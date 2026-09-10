import time
import json
import joblib
import torch
import librosa
import numpy as np
import pandas as pd
from signalflow import *
from pynput.keyboard import Key, Listener, KeyCode

import dataProcess
from dataOnsetDetect import OnsetDetection
from transformerT2V import TransformerDiscreteT2V



def makeCrownEnvelope(dur_samples, fade_perc=0.01):
	# make crown envelope
	crown_window = np.ones(dur_samples)
	fade_perc = 0.01 # percentage of fade in/out
	fade_num_samples = int(fade_perc * dur_samples)
	fade_in = np.linspace(0,1,fade_num_samples)
	fade_out = np.linspace(1,0,fade_num_samples)
	crown_window[0:fade_num_samples] = fade_in
	crown_window[crown_window.shape[0]-fade_num_samples:] = fade_out
	return crown_window



if __name__ == "__main__":

	# # Collect events until released
	# with Listener(on_press=on_press,on_release=on_release) as listener:
	# 	listener.join()

	# LOAD PARAMS
	date_folder_dir = '2026-02-04'
	model_name = '1770195602-transformerDiscreteT2V'
	model_dir = f'01_model_logs/{date_folder_dir}/{model_name}'
	sound_save_filename = f'02_output_sessions/output-{model_name}_{date_folder_dir}.wav'
	playimpulse = False

	with open(f'{model_dir}/featureExtraction_params.json', 'r') as f:
		processing_params = json.load(f)
	with open(f'{model_dir}/training_config.json', 'r') as f:
		training_parameters = json.load(f)
	model_load_path = f'{model_dir}/best_model.pth'

	# RENDER TRACKS TO AUDIO AND VISUALIZE
	sr = processing_params['sample_rate']
	window_size = processing_params['FFT_window_size']
	hop_size = processing_params['hop_size']
	num_MFCC = processing_params['num_MFCC']
	num_chroma = processing_params['num_chroma']

	# LOAD TARGET CORPUS SOUNDFILES FOR CONCATENATIVE SYNTH
	target_corpus_df = pd.read_csv(f'{training_parameters['target_corpus_path']}/corpus_discrete.csv', index_col=0)
	unique_filenames = unique_elements = list(set(target_corpus_df['filename'].values))
	concatenative_corpus = {}
	for audiofilepath in unique_filenames:
		signal, _ = librosa.load(audiofilepath, sr=sr, mono=True)
		concatenative_corpus[audiofilepath] = signal

	# get feat names from inputs
	features_source, _ = dataProcess.getFeatureNames(training_parameters['features_source'], 
													mfcc_N=processing_params['num_MFCC'], 
													chroma_N=processing_params['num_chroma'],
													includeStd=training_parameters['includeStd_src'])
	features_target, _ = dataProcess.getFeatureNames(training_parameters['features_target'], 
													mfcc_N=processing_params['num_MFCC'], 
													chroma_N=processing_params['num_chroma'],
													includeStd=training_parameters['includeStd_tgt'])
	all_feat_names = ['rms', 'pitch', 'cent', 'flatness', 'rolloff', 'MFCC', 'chroma']
	all_features, _ = dataProcess.getFeatureNames(all_feat_names, 
													mfcc_N=processing_params['num_MFCC'], 
													chroma_N=processing_params['num_chroma'],
													includeStd=True)


	# create model
	model = TransformerDiscreteT2V(
			num_featuresin=len(features_source)+1,
			num_featuresout=len(features_target)+1,
			num_frequency=training_parameters['num_frequency'],
			max_seq_len=training_parameters['max_sequence_length'],
			dim_model=training_parameters['dim_model'], 
			num_heads=training_parameters['num_heads'], 
			num_encoder_layers=training_parameters['num_encoder_layers'], 
			num_decoder_layers=training_parameters['num_decoder_layers'], 
			dim_feedforward=training_parameters['dim_feedforward'], 
			dropout_p=training_parameters['dropout_p']
	).to('cpu')

	model.load_state_dict(torch.load(model_load_path, weights_only=True))
	model.eval()
	src_scaler = joblib.load(f'{model_dir}/source_scaler.pkl')
	tgt_scaler = joblib.load(f'{model_dir}/target_scaler.pkl')
	src_time_scaler = joblib.load(f'{model_dir}/source_time_scaler.pkl')
	tgt_time_scaler = joblib.load(f'{model_dir}/target_time_scaler.pkl')


	# START AUDIO GRAPH
	sr = processing_params['sample_rate']
	window_size = processing_params['FFT_window_size']
	hop_size = processing_params['hop_size']

	config = AudioGraphConfig()
	config.sample_rate = sr
	config.output_buffer_size = window_size
	config.output_device_name = "UltraLite-mk5"
	config.input_device_name = "BlackHole 16ch"
	# config.output_device_name = "UMC202HD 192k"
	# config.input_device_name = "UMC202HD 192k"
	graph = AudioGraph(config)
	# graph.poll(2)
	audio_in = AudioIn() 
	onsetDetector = OnsetDetection(sr=sr)
	impulse = Impulse(frequency=0)
	graph.play([audio_in*4, impulse]) 


	# RECORD OUTPUT
	recording = False
	def on_press(key):
		global recording
		global sound_save_filename
		# print('{0} pressed'.format(key))
		if key == KeyCode.from_char('s'):
			if not recording:
				print('STARTING RECORDING')
				graph.start_recording(sound_save_filename)
				recording = True
			else:
				print('STOPPING RECORDING AND SAVING...')
				graph.stop_recording()
				recording = False

	def on_release(key):
		return

	# Create a non-blocking listener
	listener = Listener(
		on_press=on_press,
		on_release=on_release)
	# Start the listener thread
	listener.start()


	# INTIALIZE AGENT
	update_interval_s = hop_size / sr
	input_buffer = Buffer(1, window_size)
	min_feedback_writer_delay = graph.output_buffer_size / graph.sample_rate # samples
	graph.add_node(FeedbackBufferWriter(input_buffer, audio_in, min_feedback_writer_delay))

	input_sound = input_buffer.data[0,:]
	prev_onset_time = time.time()
	start_time = prev_onset_time

	WINDOW_N_SAMPLES = training_parameters['window_size_s'] * processing_params['sample_rate']

	# predict output 
	input_src_array = np.zeros((1,len(features_source)+1))
	input_tgt_array = np.zeros((1,len(features_target)+1))

	features = np.zeros(len(features_source))
	last_onset = 0
	PRED_TIME_SAMPLES = 0
	pred_signal_sample = 0
	src_signal_sample = 0
	model_inference_time_s = 0.003
	inference_time_samples = model_inference_time_s * sr
	current_src_onset = np.zeros(window_size)
	current_tgt_onset = np.zeros(window_size)
	tgt_samples_left = 0
	processing_params['silence_threshold'] = 0.01 # SILENCE THREHSOLD NEEDS TO BE ADAPTED TO LIVE SITUATION

	# log events
	src_past_events = []
	src_past_events_starttimes = []
	tgt_past_events = []
	tgt_past_events_starttimes = []
	device = 'cpu'
	while True:
		signal = input_buffer.data[0,:]
		# LISTEN TO INPUT (detect onset in current window)
		onsets_samples = onsetDetector(signal.reshape(1,-1))
		if onsets_samples:
			if onsets_samples[0] != last_onset:
				# print('NEW ONSET!')
				print('ONSET')
				print(onsets_samples)

				# analyze last onset
				current_src_onset = np.concatenate((current_src_onset, signal[:onsets_samples[0]]))
				print(current_src_onset.shape)
				# compute features for last onset
				featurestot = dataProcess.computeFeaturesOnset(current_src_onset, 
																sr, 
																window_size, 
																hop_size, 
																mfcc_N=processing_params['num_MFCC'], 
																chroma_N=processing_params['num_chroma'],
																max_length_samples=processing_params['max_event_duration_samples'])
				# detect rest here
				featurestot[-1] = current_src_onset.shape[0] # change duration to detected one (because of slow window)
				features_df = pd.DataFrame(data=np.array(featurestot).reshape(1,-1), columns=all_features)
				features = features_df[features_source].values

				# append event to event log
				src_past_events.append(features.reshape(-1))
				src_past_events_starttimes.append(src_signal_sample)
				src_signal_sample += int(current_src_onset.shape[0])
				# print(f'NEW ONSET --> amp: {features.reshape(-1)[0]:.3f} duration: {features.reshape(-1)[-1]:.3f}')

				# start new onset
				current_src_onset = signal[onsets_samples[0]:]
				last_onset = onsets_samples[0]
				if playimpulse:
					graph.play(StereoPanner(Impulse(frequency=0), -1))
 
		else:
			# current_src_onset = np.concatenate((current_src_onset, signal[hop_size:]))

			# check if src onset is finished and a rest has to be added
			rms = librosa.feature.rms(y=signal, frame_length=window_size, hop_length=hop_size)
			if rms[0][:32].mean() < processing_params['silence_threshold']:
				# we are inside a rest

				# check if max rest size has been reached
				if current_src_onset.shape[0] > processing_params['max_event_duration_samples']:
					# print('REST ENDED, NEW REST BEGINS')
					print('REST')
					features = np.zeros(len(features_source)) # it's a rest so features are all zero
					features[-1] = processing_params['max_event_duration_samples']

					# append event to event log
					src_past_events.append(features.reshape(-1))
					src_past_events_starttimes.append(src_signal_sample)
					src_signal_sample += int(processing_params['max_event_duration_samples'])

					print(processing_params['max_event_duration_samples'])
					# print(f'NEW REST --> amp: {features.reshape(-1)[0]:.3f} duration: {features.reshape(-1)[-1]:.3f}')
					current_src_onset = np.zeros(signal[hop_size:].shape)
				else:
					# print('IN A REST')
					current_src_onset = np.concatenate((current_src_onset, np.zeros(signal[current_src_onset.shape[0]-processing_params['max_event_duration_samples']:].shape)))
			else:
				# we are inside an onset, need to check if it ends
				end_event_sample = -1
				for k, rms_value in enumerate(rms[0].tolist()):
					if rms_value > processing_params['silence_threshold']:
						end_event_sample=k
						break
				if end_event_sample != -1:
					# print('ONSET ENDED, STARTING A REST')
					print('REST')
					# analyze last onset
					current_src_onset = np.concatenate((current_src_onset, signal[:end_event_sample]))
					# compute features for last onset
					print(current_src_onset.shape)
					featurestot = dataProcess.computeFeaturesOnset(current_src_onset, 
																	sr, 
																	window_size, 
																	hop_size, 
																	mfcc_N=processing_params['num_MFCC'], 
																	chroma_N=processing_params['num_chroma'],
																	max_length_samples=processing_params['max_event_duration_samples'])
					# detect rest here
					featurestot[-1] = current_src_onset.shape[0] # change duration to detected one (because of slow window)
					features_df = pd.DataFrame(data=np.array(featurestot).reshape(1,-1), columns=all_features)
					features = features_df[features_source].values

					# append event to event log
					src_past_events.append(features.reshape(-1))
					src_past_events_starttimes.append(src_signal_sample)
					src_signal_sample += int(processing_params['max_event_duration_samples'])

					# print(f'NEW ONSET --> amp: {features.reshape(-1)[0]:.3f} duration: {features.reshape(-1)[-1]:.3f}')
					current_src_onset = np.zeros(signal[hop_size:].shape)
				else:
					# print('IN AN ONSET')
					current_src_onset = np.concatenate((current_src_onset, signal[hop_size:]))


		# CHECK IF IT IS TIME TO GENERATE NEW OUTPUT (detect onset in current window)
		if tgt_samples_left - inference_time_samples < window_size:

			# COMPUTE RESPONSE
			input_src_array_norm = np.zeros_like(input_src_array)
			input_tgt_array_norm = np.zeros_like(input_tgt_array)
			input_src_array_norm[:,:-1] = src_scaler.transform(input_src_array[:,:-1])
			input_tgt_array_norm[:,:-1] = tgt_scaler.transform(input_tgt_array[:,:-1])
			input_src_array_norm[:,-1] = src_time_scaler.transform(input_src_array[:,-1].reshape(-1,1)).reshape(-1)
			input_tgt_array_norm[:,-1] = tgt_time_scaler.transform(input_tgt_array[:,-1].reshape(-1,1)).reshape(-1)
			ones_src = np.ones((training_parameters['max_sequence_length'],len(features_source)+1)) * training_parameters['pad_token']
			ones_tgt = np.ones((training_parameters['max_sequence_length'],len(features_target)+1)) * training_parameters['pad_token']
			ones_src[:input_src_array_norm.shape[0],:] = input_src_array_norm
			ones_tgt[:input_tgt_array_norm.shape[0],:] = input_tgt_array_norm
			input_src_array_norm = ones_src
			input_tgt_array_norm = ones_tgt
			input_src_array_norm = torch.tensor(input_src_array_norm.reshape(1, input_src_array_norm.shape[0], input_src_array_norm.shape[1])).type(torch.FloatTensor).to(device)
			input_tgt_array_norm = torch.tensor(input_tgt_array_norm.reshape(1, input_tgt_array_norm.shape[0], input_tgt_array_norm.shape[1])).type(torch.FloatTensor).to(device)
			start_model_time = time.time()
			src_pad_mask = model.create_pad_mask(input_src_array_norm, pad_token=training_parameters['pad_token']).to(device)
			tgt_pad_mask = model.create_pad_mask(input_tgt_array_norm, pad_token=training_parameters['pad_token']).to(device)
			tgt_mask = model.get_tgt_mask(input_tgt_array_norm.size(1)).to(device)
			y = model(input_src_array_norm, input_tgt_array_norm, tgt_mask=tgt_mask, src_pad_mask=src_pad_mask[:,:,0], tgt_pad_mask=tgt_pad_mask[:,:,0]).detach().cpu().numpy()
			y_denorm = tgt_scaler.inverse_transform(y.reshape(y.shape[0], y.shape[2]))
			next_token = y_denorm[-1,:]

			# retreive sample and play
			closest_index = (target_corpus_df[features_target] - next_token).abs().idxmin()
			closest_row = target_corpus_df.loc[closest_index[0]]
			sound_segment = concatenative_corpus[closest_row['filename']][int(closest_row['event_start']):int(closest_row['event_start'])+int(closest_row['event_duration'])]
			crown_window = makeCrownEnvelope(int(closest_row['event_duration']), fade_perc=0.01)
			buf = Buffer(sound_segment * crown_window)
			player = BufferPlayer(buf)
			delay_s = tgt_samples_left / sr
			graph.play(StereoPanner(OneTapDelay(input=player, delay_time=delay_s), 1))

			# append event to event log
			tgt_past_events.append(next_token.reshape(-1))
			tgt_past_events_starttimes.append(pred_signal_sample)
			# tgt_past_events = tgt_past_events[:-training_parameters['max_sequence_length']]
			# tgt_past_events_starttimes = tgt_past_events_starttimes[:-training_parameters['max_sequence_length']]
			pred_signal_sample += int(closest_row['event_duration'])

			if len(src_past_events) > 0 and len(tgt_past_events) > 0:

				# get source events in window before current
				input_src = np.array(src_past_events)
				times_src = np.array(src_past_events_starttimes)#.reshape(-1,1)
				window_start_idx = (pred_signal_sample - PRED_TIME_SAMPLES) - WINDOW_N_SAMPLES
				indices = np.where((times_src >= window_start_idx) & (times_src < pred_signal_sample))
				x_src_window_times = (pred_signal_sample - PRED_TIME_SAMPLES) - times_src[indices].reshape(-1,1) # add time difference index
				input_src_array = np.concatenate((input_src[indices], x_src_window_times.reshape(-1,1)), axis=1)[-training_parameters['max_sequence_length']:,:]

				# # remove old events from memory
				# src_past_events_starttimes = times_src[indices].tolist()
				# src_past_events = input_src[indices].tolist()

				# update sequences
				input_tgt = np.array(tgt_past_events)
				times_tgt = np.array(tgt_past_events_starttimes)#.reshape(-1,1)
				indices = np.where((times_tgt >= window_start_idx) & (times_tgt < pred_signal_sample))
				x_tgt_window_times = (pred_signal_sample - PRED_TIME_SAMPLES) - times_tgt[indices].reshape(-1,1) # add time difference index
				input_tgt_array = np.concatenate((input_tgt[indices], x_tgt_window_times.reshape(-1,1)), axis=1)[-training_parameters['max_sequence_length']:,:]
				
				# # remove old events from memory
				# tgt_past_events_starttimes = times_tgt[indices].tolist()
				# tgt_past_events = input_tgt[indices].tolist()

			# samples countdown
			tgt_samples_left = sound_segment.shape[0] - (window_size - (tgt_samples_left - inference_time_samples))
		else:
			tgt_samples_left -= hop_size 

		# check every 
		time.sleep(update_interval_s)

	graph.wait()



