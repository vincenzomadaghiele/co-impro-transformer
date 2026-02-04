import os
import time
import json
import torch
import joblib 
import random
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler

import dataVisualize
import dataProcess
from transformerDiscreteT2V import TransformerDiscreteT2V
from transformerDiscreteT2V_train import batchify, validationLoop, spectral_loss, evaluateMrSTFT, evaluateAccuracyWithClustering


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


def renderModelOutput(model_dir, eval_track_names):

	# LOAD PARAMS
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

	# EVALUATE TRACKS
	for eval_track_name in eval_track_names:
		print()
		print(f'Generating concatenative rending for track {eval_track_name}...')
		print('-'*50)
		print()
		# extract features from source track
		src_audiofilepath = f'{training_parameters['sound_corpus_path']}/{eval_track_name}/{processing_params['source_track_name']}.wav'
		src_signal, _ = librosa.load(src_audiofilepath, sr=sr, mono=True)
		pred_signal = np.zeros(src_signal.shape[0])
		signal_df, _ = dataProcess.extractFeatures(src_signal, 
													sr, 
													window_size, 
													hop_size, 
													silence_threshold=processing_params['silence_threshold'],
													mfcc_N=num_MFCC, 
													chroma_N=num_chroma,
													max_event_duration_samples=processing_params['max_event_duration_samples'], 
													min_event_duration_samples=processing_params['min_event_duration_samples'])

		print(signal_df.head())

		# normalize for input
		input_src = signal_df[features_source].values
		times_src = signal_df['event_start'].values

		WINDOW_N_SAMPLES = training_parameters['window_size_s'] * processing_params['sample_rate']

		# predict output 
		input_src_array = np.zeros((1,len(features_source)+1))
		input_tgt_array = np.zeros((1,len(features_target)+1))

		# predict each time a new target token is finished (or tot tokens in advace)
		pred_signal_sample = 0
		PRED_TIME_SAMPLES = 0
		# for keeping track of previous events
		# src_past_events = []
		# src_past_events_starttimes = []
		tgt_past_events = []
		tgt_past_events_starttimes = []

		# for stats
		inference_times = []
		model_inference_times = []
		table_lookup_times = []
		sequence_update_times = []
		device = 'cpu'
		while pred_signal_sample < src_signal.shape[0]:

			start_inf_time = time.time()
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
			# input_src_array_norm = torch.tensor(input_src_array_norm.reshape(input_src_array_norm.shape[0], 1, input_src_array_norm.shape[1])).type(torch.FloatTensor).to(device)
			# input_tgt_array_norm = torch.tensor(input_tgt_array_norm.reshape(input_tgt_array_norm.shape[0], 1, input_tgt_array_norm.shape[1])).type(torch.FloatTensor).to(device)
			input_src_array_norm = torch.tensor(input_src_array_norm.reshape(1, input_src_array_norm.shape[0], input_src_array_norm.shape[1])).type(torch.FloatTensor).to(device)
			input_tgt_array_norm = torch.tensor(input_tgt_array_norm.reshape(1, input_tgt_array_norm.shape[0], input_tgt_array_norm.shape[1])).type(torch.FloatTensor).to(device)
			start_model_time = time.time()
			src_pad_mask = model.create_pad_mask(input_src_array_norm, pad_token=training_parameters['pad_token']).to(device)
			tgt_pad_mask = model.create_pad_mask(input_tgt_array_norm, pad_token=training_parameters['pad_token']).to(device)
			tgt_mask = model.get_tgt_mask(input_tgt_array_norm.size(1)).to(device)
			y = model(input_src_array_norm, input_tgt_array_norm, tgt_mask=tgt_mask, src_pad_mask=src_pad_mask[:,:,0], tgt_pad_mask=tgt_pad_mask[:,:,0]).detach().cpu().numpy()
			end_model_time = time.time()
			y_denorm = tgt_scaler.inverse_transform(y.reshape(y.shape[0], y.shape[2]))
			next_token = y_denorm[-1,:]
			# print(f'{pred_signal_sample}: {next_token}')

			# retreive sound segment
			start_lookup_time = time.time()
			closest_index = (target_corpus_df[features_target] - next_token).abs().idxmin()
			closest_row = target_corpus_df.loc[closest_index[0]]
			sound_segment = concatenative_corpus[closest_row['filename']][int(closest_row['event_start']):int(closest_row['event_start'])+int(closest_row['event_duration'])]
			end_lookup_time = time.time()

			# update model inputs at time of end of last predicted token
			tgt_past_events.append(next_token)
			tgt_past_events_starttimes.append(pred_signal_sample)

			# update input sequences
			if pred_signal_sample + int(closest_row['event_duration']) < src_signal.shape[0]:
				start_sequpdate_time = time.time()
				# compute concatenative synthesis
				crown_window = makeCrownEnvelope(int(closest_row['event_duration']), fade_perc=0.01)
				pred_signal[int(pred_signal_sample):int(pred_signal_sample)+int(closest_row['event_duration'])] = sound_segment * crown_window
				pred_signal_sample += int(closest_row['event_duration'])


				# get source events in window before current
				window_start_idx = (pred_signal_sample - PRED_TIME_SAMPLES) - WINDOW_N_SAMPLES
				indices = np.where((times_src >= window_start_idx) & (times_src < pred_signal_sample))
				# add time difference index
				x_src_window_times = (pred_signal_sample - PRED_TIME_SAMPLES) - times_src[indices]
				input_src_array = np.concatenate((input_src[indices], x_src_window_times.reshape(-1,1)), axis=1)[-training_parameters['max_sequence_length']:,:]


				times_tgt = np.array(tgt_past_events_starttimes)
				input_tgt = np.array(tgt_past_events)
				indices = np.where((times_tgt >= window_start_idx) & (times_tgt < pred_signal_sample))
				# add time difference index
				x_tgt_window_times = (pred_signal_sample - PRED_TIME_SAMPLES) - times_tgt[indices]
				input_tgt_array = np.concatenate((input_tgt[indices], x_tgt_window_times.reshape(-1,1)), axis=1)[-training_parameters['max_sequence_length']:,:]

				end_sequpdate_time = time.time()

				end_inf_time = time.time()
				inference_times.append(end_inf_time-start_inf_time)
				model_inference_times.append(end_model_time-start_model_time)
				table_lookup_times.append(end_lookup_time-start_lookup_time)
				sequence_update_times.append(end_sequpdate_time-start_sequpdate_time)
			else:
				break



		print()
		print(f'Mean total inference time: {np.array(inference_times).mean():.6f}')
		print(f'Mean model inference time: {np.array(model_inference_times).mean():.6f}')
		print(f'Mean table lookup time: {np.array(table_lookup_times).mean():.6f}')
		print(f'Mean sequence update time: {np.array(sequence_update_times).mean():.6f}')
		print()
		print('Exporting...')

		# EXPORT AUDIOFILES
		results_dir = f'{model_dir}/00_results/{eval_track_name}'
		os.makedirs(results_dir, exist_ok=True)
		out_signal = np.concatenate((src_signal.reshape(-1,1), pred_signal.reshape(-1,1)), axis=1)
		sf.write(f'{results_dir}/{eval_track_name}_combined.wav', out_signal, sr, subtype='PCM_24')
		sf.write(f'{results_dir}/{eval_track_name}_{processing_params['source_track_name']}.wav', src_signal, sr, subtype='PCM_24')
		sf.write(f'{results_dir}/{eval_track_name}_{processing_params['target_track_name']}.wav', pred_signal, sr, subtype='PCM_24')


		# VISUALIZE
		src_signal_path = f'{results_dir}/{eval_track_name}_{processing_params['source_track_name']}.wav'
		src_signal, _ = librosa.load(src_signal_path, sr=sr, mono=True)
		src_signal_df, _ = dataProcess.extractFeatures(src_signal, 
														sr, 
														window_size, 
														hop_size, 
														0.001, 
														num_MFCC, 
														num_chroma, 
														True, 
														processing_params['max_event_duration_samples'], 
														processing_params['min_event_duration_samples'])
		src_signal_df['filename'] = src_signal_path
		src_signal_df.to_csv(f'{results_dir}/src_corpus.csv')

		tgt_signal_path = f'{results_dir}/{eval_track_name}_{processing_params['target_track_name']}.wav'
		tgt_signal, _ = librosa.load(tgt_signal_path, sr=sr, mono=True)
		tgt_signal_df, _ = dataProcess.extractFeatures(tgt_signal, 
														sr, window_size, 
														hop_size, 
														0.001, 
														num_MFCC, 
														num_chroma, 
														True, 
														processing_params['max_event_duration_samples'], 
														processing_params['min_event_duration_samples'])
		tgt_signal_df['filename'] = tgt_signal_path
		tgt_signal_df.to_csv(f'{results_dir}/tgt_corpus.csv')

		print('Computing visualizations...')
		dataVisualize.plotCoupleFromDB(f'{results_dir}/src_corpus.csv', 
									f'{results_dir}/tgt_corpus.csv', 
									src_signal_path,
									tgt_signal_path,
									sample_rate=sr,
									plot_feature_src='rms_mean', 
									plot_feature_tgt='rms_mean', 
									sample_start_s=0, 
									sample_duration_s=None,
									save_dir=f'{results_dir}/bass_drums')
		dataVisualize.plotCoupleFromDB(f'{results_dir}/src_corpus.csv', 
									f'{results_dir}/tgt_corpus.csv', 
									src_signal_path,
									tgt_signal_path,
									sample_rate=sr,
									plot_feature_src='rms_mean', 
									plot_feature_tgt='rms_mean', 
									sample_start_s=20, 
									sample_duration_s=30,
									save_dir=f'{results_dir}/bass_drums_short')



def evaluateModel(model_dir):
	# LOAD PARAMS
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
	# SET SEED
	torch.manual_seed(training_parameters['seed'])
	random.seed(training_parameters['seed'])
	np.random.seed(training_parameters['seed'])

	src_scaler = joblib.load(f'{model_dir}/source_scaler.pkl')
	tgt_scaler = joblib.load(f'{model_dir}/target_scaler.pkl')
	src_time_scaler = joblib.load(f'{model_dir}/source_time_scaler.pkl')
	tgt_time_scaler = joblib.load(f'{model_dir}/target_time_scaler.pkl')


	# LOAD CORPUS
	source_corpus_df = pd.read_csv(f'{training_parameters['source_corpus_path']}/corpus_discrete.csv', index_col=0)
	target_corpus_df = pd.read_csv(f'{training_parameters['target_corpus_path']}/corpus_discrete.csv', index_col=0)
	src_signals_dfs = []
	tgt_signals_dfs = []
	unique_src_filenames = source_corpus_df['filename'].unique()
	unique_tgt_filenames = target_corpus_df['filename'].unique()
	unique_src_filenames = [name.split('/')[-2] for name in unique_src_filenames]
	unique_tgt_filenames = [name.split('/')[-2] for name in unique_tgt_filenames]
	common_filenames = list(set(unique_src_filenames) & set(unique_tgt_filenames))
	# common_filenames = common_filenames[:4] # select a subset of data
	for filename in common_filenames:
		# assume that audio tracks are located one folder before datasets
		partial_src_df = source_corpus_df[source_corpus_df['filename'] == f'{training_parameters['sound_corpus_path']}/{filename}/{processing_params['source_track_name']}.wav']
		partial_tgt_df = target_corpus_df[target_corpus_df['filename'] == f'{training_parameters['sound_corpus_path']}/{filename}/{processing_params['target_track_name']}.wav']
		# check that both are non-empty
		if partial_src_df.shape[0] > 0 and partial_tgt_df.shape[0] > 0:
			src_signals_dfs.append(partial_src_df)
			tgt_signals_dfs.append(partial_tgt_df)
		# src_signals_dfs.append(source_corpus_df[source_corpus_df['filename'] == f'{training_parameters['sound_corpus_path']}/{filename}/{src_processing_params['source_track_name']}.wav'])
		# tgt_signals_dfs.append(target_corpus_df[target_corpus_df['filename'] == f'{training_parameters['sound_corpus_path']}/{filename}/{src_processing_params['target_track_name']}.wav'])

	print(source_corpus_df.head())
	# print(source_corpus_df.columns)

	# normalize source features
	src_features = source_corpus_df[features_source].values
	# src_scaler = StandardScaler()
	# src_scaler.fit(src_features)
	src_normalized_sequences = []
	for seq_df in src_signals_dfs:
		sequence_df = pd.DataFrame(data=src_scaler.transform(seq_df[features_source].values), columns=features_source)
		sequence_df['event_start'] = seq_df['event_start'].values
		src_normalized_sequences.append(sequence_df)

	# normalize target features
	tgt_features = target_corpus_df[features_target].values
	# tgt_scaler = StandardScaler()
	# tgt_scaler.fit(tgt_features)
	tgt_normalized_sequences = []
	for seq_df in tgt_signals_dfs:
		sequence_df = pd.DataFrame(data=tgt_scaler.transform(seq_df[features_target].values), columns=features_target)
		sequence_df['event_start'] = seq_df['event_start'].values
		tgt_normalized_sequences.append(sequence_df)


	# make dataloader
	window_size_s = training_parameters['window_size_s']
	pred_time_s = training_parameters['pred_time_s']
	WINDOW_N_SAMPLES = window_size_s * processing_params['sample_rate']
	PRED_TIME_SAMPLES = pred_time_s * processing_params['sample_rate']
	data = []
	# max_seq_len = 0
	seq_lens = []
	all_src_times = []
	all_tgt_times = []
	for k in range(len(src_normalized_sequences)):
		X_values = src_normalized_sequences[k][features_source].values
		y_values = tgt_normalized_sequences[k][features_target].values
		X_times = src_normalized_sequences[k]['event_start'].values
		y_times = tgt_normalized_sequences[k]['event_start'].values
		for j in range(y_times.shape[0]-1):
			y_tgt = y_values[j+1,:]
			# we subtract inference time 
			# so that we can call inference some time before the end of an onset
			window_start_idx = (y_times[j+1] - PRED_TIME_SAMPLES) - WINDOW_N_SAMPLES
			indices = np.where((X_times >= window_start_idx) & (X_times < y_times[j+1]))
			# add time difference index
			x_src_window_times = (y_times[j+1] - PRED_TIME_SAMPLES) - X_times[indices]
			x_src = np.concatenate((X_values[indices], x_src_window_times.reshape(-1,1)), axis=1)
			all_src_times += x_src_window_times.tolist()
			indices = np.where((y_times >= window_start_idx) & (y_times < y_times[j+1]))
			# add time difference index
			x_tgt_window_times = (y_times[j+1] - PRED_TIME_SAMPLES) - y_times[indices]
			x_tgt = np.concatenate((y_values[indices], x_tgt_window_times.reshape(-1,1)), axis=1)
			all_tgt_times += x_tgt_window_times.tolist()
			data.append([x_src, x_tgt, y_tgt])
			seq_lens.append(x_src.shape[0])
			seq_lens.append(x_tgt.shape[0])


	# MAX_SEQ_LEN = int(np.array(seq_lens).mean() + np.array(seq_lens).std())
	# if model_load_path:
	MAX_SEQ_LEN = training_parameters['max_sequence_length']
	# remove rows where time is less than min or 
	# normalize time feature
	# src_time_scaler = StandardScaler()
	# src_time_scaler.fit(np.array(all_src_times).reshape(-1,1))
	# tgt_time_scaler = StandardScaler()
	# tgt_time_scaler.fit(np.array(all_tgt_times).reshape(-1,1))
	new_data = []
	for datum in data:
		# print(datum[0].shape)
		x_src = datum[0]
		x_src[:,-1] = src_time_scaler.transform(x_src[:,-1].reshape(-1,1)).reshape(-1)
		x_tgt = datum[1]
		x_tgt[:,-1] = src_time_scaler.transform(x_tgt[:,-1].reshape(-1,1)).reshape(-1)
		x_src = x_src[-MAX_SEQ_LEN:,:]
		x_tgt = x_tgt[-MAX_SEQ_LEN:,:]
		new_data.append([x_src, x_tgt, datum[2]])
	data = new_data


	# PAD SEQUENCES
	PAD_TOKEN = training_parameters['pad_token']
	new_data = []
	for datum in data:
		x_src = np.ones((MAX_SEQ_LEN, len(features_source)+1)) * PAD_TOKEN
		x_tgt = np.ones((MAX_SEQ_LEN, len(features_target)+1)) * PAD_TOKEN
		x_src[:datum[0].shape[0],:] = datum[0]
		x_tgt[:datum[1].shape[0],:] = datum[1]
		new_data.append([x_src, x_tgt, datum[2]])
	data = new_data

	# with open(f'{logdir}/training_config.json', 'r') as f:
	# 	training_log = json.load(f)
	# training_log['max_sequence_length'] = MAX_SEQ_LEN
	# with open(f'{logdir}/training_config.json', 'w', encoding='utf-8') as f:
	# 	json.dump(training_log, f, ensure_ascii=False, indent=4)

	# shuffle sequences
	np.random.shuffle(data)

	# train-validation split
	train_perc = training_parameters['dataset_training_percentage']
	val_perc = training_parameters['dataset_validation_percentage']
	train_data = data[:int(len(data)*train_perc)]
	test_data = data[int(len(data)*train_perc):]
	val_data = train_data[int(len(train_data)*(1-val_perc)):]
	train_data = train_data[:int(len(train_data)*(1-val_perc))]

	# make training batches
	batch_size = training_parameters['batch_size']
	train_dataloader = batchify(train_data, batch_size)
	val_dataloader = batchify(val_data, batch_size)
	test_dataloader = batchify(test_data, batch_size)

	# find device
	device = torch.device("cpu")
	if torch.cuda.is_available():
		device = torch.device("cuda")
	elif torch.backends.mps.is_available():
		device = torch.device("mps")

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
	).to(device)

	model.load_state_dict(torch.load(model_load_path, weights_only=True))
	model.eval()

	opt = torch.optim.SGD(model.parameters(), lr=training_parameters['learning_rate'])
	if training_parameters['loss_fn'] == 'L1':
		loss_fn = nn.L1Loss()
	else:
		loss_fn = nn.MSELoss()

	evaluation_metrics = {}

	train_loss = validationLoop(model, train_dataloader, loss_fn, opt, device, training_parameters['pad_token'])
	evaluation_metrics['training_MSE'] = train_loss
	validation_loss = validationLoop(model, val_dataloader, loss_fn, opt, device, training_parameters['pad_token'])
	evaluation_metrics['validation_MSE'] = validation_loss
	test_loss = validationLoop(model, test_dataloader, loss_fn, opt, device, training_parameters['pad_token'])
	evaluation_metrics['test_MSE'] = test_loss


	# LOAD TARGET CORPUS SOUNDFILES FOR CONCATENATIVE SYNTH
	target_corpus_df = pd.read_csv(f'{training_parameters['target_corpus_path']}/corpus_discrete.csv', index_col=0)
	unique_filenames = unique_elements = list(set(target_corpus_df['filename'].values))
	concatenative_corpus = {}
	for audiofilepath in unique_filenames:
		signal, _ = librosa.load(audiofilepath, sr=processing_params['sample_rate'], mono=True)
		concatenative_corpus[audiofilepath] = signal

	train_mrSTFT, train_r2 = evaluateMrSTFT(model, train_dataloader, device, PAD_TOKEN, 
										tgt_scaler, target_corpus_df, features_target, concatenative_corpus)
	val_mrSTFT, val_r2 = evaluateMrSTFT(model, val_dataloader, device, PAD_TOKEN, 
										tgt_scaler, target_corpus_df, features_target, concatenative_corpus)
	test_mrSTFT, test_r2 = evaluateMrSTFT(model, test_dataloader, device, PAD_TOKEN, 
										tgt_scaler, target_corpus_df, features_target, concatenative_corpus)
	evaluation_metrics['training_mrSTFT'] = train_mrSTFT
	evaluation_metrics['validation_mrSTFT'] = val_mrSTFT
	evaluation_metrics['test_mrSTFT'] = test_mrSTFT
	evaluation_metrics['train_r2'] = train_r2
	evaluation_metrics['val_r2'] = val_r2
	evaluation_metrics['test_r2'] = test_r2

	#### TEST ACCURACY WITH CLUSTERING
	try:
		tgt_brc = joblib.load(f'{training_parameters['target_corpus_path']}/tgt_birch_classifier.pkl')
		train_accuracy, train_f1 = evaluateAccuracyWithClustering(model, train_dataloader, device, 
														PAD_TOKEN, tgt_scaler, target_corpus_df, 
														features_target, tgt_brc)
		validation_accuracy, val_f1 = evaluateAccuracyWithClustering(model, val_dataloader, device, 
														PAD_TOKEN, tgt_scaler, target_corpus_df, 
														features_target, tgt_brc)
		test_accuracy, test_f1 = evaluateAccuracyWithClustering(model, test_dataloader, device, 
														PAD_TOKEN, tgt_scaler, target_corpus_df, 
														features_target, tgt_brc)
		evaluation_metrics['training accuracy'] = train_accuracy
		evaluation_metrics['validation accuracy'] = validation_accuracy
		evaluation_metrics['test accuracy'] = test_accuracy
		evaluation_metrics['training F1'] = train_f1
		evaluation_metrics['validation F1'] = val_f1
		evaluation_metrics['test F1'] = test_f1
	except:
		print("Couldn't evaluate clustering accuracy")

	# with open(f'{logdir}/evaluation.json', 'w', encoding='utf-8') as f:
	# 	json.dump(evaluation_metrics, f, ensure_ascii=False, indent=4)

	print()
	print('MODEL EVALUATION')
	print('-'*20)
	print(f'training MSE: {evaluation_metrics['training_MSE']}')
	print(f'validation MSE: {evaluation_metrics['validation_MSE']}')
	print(f'test MSE: {evaluation_metrics['test_MSE']}')
	print(f'training R2: {evaluation_metrics['train_r2']}')
	print(f'validation R2: {evaluation_metrics['val_r2']}')
	print(f'test R2: {evaluation_metrics['test_r2']}')
	print(f'training mrSTFT: {evaluation_metrics['training_mrSTFT']}')
	print(f'validation mrSTFT: {evaluation_metrics['validation_mrSTFT']}')
	print(f'test mrSTFT: {evaluation_metrics['validation_mrSTFT']}')
	try:
		print(f'training accuracy: {evaluation_metrics['training accuracy']}')
		print(f'validation accuracy: {evaluation_metrics['validation accuracy']}')
		print(f'test accuracy: {evaluation_metrics['test accuracy']}')
		print(f'training F1: {evaluation_metrics['training F1']}')
		print(f'validation F1: {evaluation_metrics['validation F1']}')
		print(f'test F1: {evaluation_metrics['test F1']}')
	except:
		print("no accuracy score")
	print('-'*20)
	print()



if __name__ == "__main__":

	# LOAD MODEL
	N_tracks = 4
	model_dir = '01_model_logs/2026-01-25/1769341178-transformerDiscreteT2V'
	with open(f'{model_dir}/training_config.json', 'r') as f:
		training_parameters = json.load(f)

	corpus_files = os.listdir(training_parameters['sound_corpus_path'])
	corpus_files = [filename for filename in corpus_files if os.path.isdir(f'{training_parameters['sound_corpus_path']}/{filename}') and filename != '00_process_src' and filename != '00_process_tgt']
	eval_track_names = corpus_files[:N_tracks]

	evaluateModel(model_dir)
	# renderModelOutput(model_dir, eval_track_names)





