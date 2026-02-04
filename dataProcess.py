import os
import json
import warnings
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dataVisualize
from dataOnsetDetect import OnsetDetection


warnings.filterwarnings("ignore")


def getFeatureNames(features, mfcc_N=13, chroma_N=12, includeStd=True):
	# features4training = ['rms', 'pitch', 'cent', 'flatness', 'rolloff', 'MFCC', 'chroma']
	fnames = [feat for feat in features if feat not in ['MFCC', 'chroma', 'event_duration']]
	feature_names = []
	feature_names_continuous = []
	for f in fnames:
		feature_names.append(f'{f}_mean')
		if includeStd:
			feature_names.append(f'{f}_std')
		feature_names_continuous.append(f)
	mfcc_N = 13
	if 'MFCC' in features:
		for i in range(mfcc_N):
			feature_names.append(f'MFCC{i}_mean')
			if includeStd:
				feature_names.append(f'MFCC{i}_std')
			feature_names_continuous.append(f'MFCC{i}')
	chroma_N = 12
	if 'chroma' in features:
		for i in range(chroma_N):
			feature_names.append(f'chroma{i}_mean')
			if includeStd:
				feature_names.append(f'chroma{i}_std')
			feature_names_continuous.append(f'chroma{i}')
	feature_names.append('event_duration')
	return feature_names, feature_names_continuous

def computeFeatures(signal, sample_rate, FFT_window_size, hop_size, mfcc_N=13, chroma_N=12):
	features = []
	rms = librosa.feature.rms(y=signal, frame_length=FFT_window_size, hop_length=hop_size)
	features.append(rms.reshape(-1))
	pitch = librosa.yin(signal, fmin=200, fmax=22000, sr=sample_rate, frame_length=FFT_window_size, hop_length=hop_size)
	features.append(pitch.reshape(-1))
	cent = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, n_fft=FFT_window_size, hop_length=hop_size)
	features.append(cent.reshape(-1))
	flatness = librosa.feature.spectral_flatness(y=signal, n_fft=FFT_window_size, hop_length=hop_size)
	features.append(flatness.reshape(-1))
	rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate, n_fft=FFT_window_size, hop_length=hop_size)
	features.append(rolloff.reshape(-1))
	mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=mfcc_N, n_fft=FFT_window_size, hop_length=hop_size)
	for mfcc_component in mfcc.tolist():
		features.append(np.array(mfcc_component).reshape(-1))
	chroma = librosa.feature.chroma_stft(y=signal, sr=sample_rate, n_chroma=chroma_N, n_fft=FFT_window_size, hop_length=hop_size)
	for chroma_component in chroma.tolist():
		features.append(np.array(chroma_component).reshape(-1))
	features = np.array(features)
	print(features.shape)
	return features

def computeFeaturesOnset(signal, sample_rate, FFT_window_size, hop_size, max_length_samples=44100, mfcc_N=13, chroma_N=12, silence_threshold=0.0005, includeStd=True, compute_rests=True):
	features = []
	rms = librosa.feature.rms(y=signal[:max_length_samples], frame_length=FFT_window_size, hop_length=hop_size)
	features.append(rms.mean())
	if includeStd:
		features.append(rms.std())
	pitch = librosa.yin(signal[:max_length_samples], fmin=100, fmax=22000, sr=sample_rate, frame_length=FFT_window_size, hop_length=hop_size)
	features.append(pitch.reshape(1,-1).mean())
	if includeStd:
		features.append(pitch.reshape(1,-1).std())
	cent = librosa.feature.spectral_centroid(y=signal[:max_length_samples], sr=sample_rate, n_fft=FFT_window_size, hop_length=hop_size)
	features.append(cent.mean())
	if includeStd:
		features.append(cent.std())
	flatness = librosa.feature.spectral_flatness(y=signal[:max_length_samples], n_fft=FFT_window_size, hop_length=hop_size)
	features.append(flatness.mean())
	if includeStd:
		features.append(flatness.std())
	rolloff = librosa.feature.spectral_rolloff(y=signal[:max_length_samples], sr=sample_rate, n_fft=FFT_window_size, hop_length=hop_size)
	features.append(rolloff.mean())
	if includeStd:
		features.append(rolloff.std())
	mfcc = librosa.feature.mfcc(y=signal[:max_length_samples], sr=sample_rate, n_mfcc=mfcc_N, n_fft=FFT_window_size, hop_length=hop_size)
	for mfcc_component in mfcc.tolist():
		features.append(np.array(mfcc_component).reshape(1,-1).mean())
		if includeStd:
			features.append(np.array(mfcc_component).reshape(1,-1).std())
	chroma = librosa.feature.chroma_stft(y=signal[:max_length_samples], sr=sample_rate, n_chroma=chroma_N, n_fft=FFT_window_size, hop_length=hop_size)
	for chroma_component in chroma.tolist():
		features.append(np.array(chroma_component).reshape(1,-1).mean())
		if includeStd:
			features.append(np.array(chroma_component).reshape(1,-1).std())
	rms = librosa.feature.rms(y=signal[:max_length_samples], frame_length=FFT_window_size, hop_length=hop_size)
	if compute_rests:
		event_duration = [1 if rms_value > silence_threshold else 0 for rms_value in rms[0].tolist()]
		event_duration = (np.array(event_duration) == 1).sum()
		features.append(librosa.frames_to_samples(event_duration, hop_length=hop_size, n_fft=FFT_window_size))
	features = np.array(features)
	return features

def computeRests(df, rest_token=0, max_length_samples=44100, min_rest_length_samples=441, last_sample_n=None):
	# max_length_samples=44100 --> 1 second
	# min_rest_length_samples=441 --> 10 ms

	new_df = pd.DataFrame(columns=df.columns)
	# check for rests at silent start
	if df.iloc[0]['event_start'] > 0:
		dur_counter = int(df.iloc[1]['event_start'])
		start_counter = 0
		while dur_counter > 0:
			if dur_counter > min_rest_length_samples:
				new_row_data = {} 
				for col in df.columns: # MAKE REST ROW
					new_row_data[col] = rest_token
				if dur_counter > max_length_samples:
					new_row_data['event_start'] = start_counter 
					new_row_data['event_duration'] = max_length_samples
					new_df = pd.concat([new_df, pd.DataFrame([new_row_data])], ignore_index=True)
					start_counter += max_length_samples
					dur_counter -= max_length_samples
				else:
					# last rest
					if dur_counter > min_rest_length_samples:
						new_row_data['event_start'] = start_counter 
						new_row_data['event_duration'] = dur_counter
						new_df = pd.concat([new_df, pd.DataFrame([new_row_data])], ignore_index=True)
					start_counter += max_length_samples
					dur_counter -= max_length_samples
			else:
				dur_counter = 0
	for i in range(len(df) - 1):
		current_row = df.iloc[i]
		next_row = df.iloc[i+1]
		new_df = pd.concat([new_df, pd.DataFrame([current_row])], ignore_index=True)
		if current_row['event_duration'] < next_row['event_start'] - current_row['event_start']:
			total_rest_duration = next_row['event_start'] - (current_row['event_start'] + current_row['event_duration'])
			start_counter = current_row['event_start'] + current_row['event_duration']
			while total_rest_duration > 0:
				if total_rest_duration > min_rest_length_samples:
					new_row_data = {} 
					for col in df.columns:
						new_row_data[col] = rest_token
					if total_rest_duration > max_length_samples:
						new_row_data['event_start'] = start_counter 
						new_row_data['event_duration'] = max_length_samples
						new_df = pd.concat([new_df, pd.DataFrame([new_row_data])], ignore_index=True)
						start_counter += max_length_samples
						total_rest_duration -= max_length_samples
					else:
						# last rest
						if total_rest_duration > min_rest_length_samples:
							new_row_data['event_start'] = start_counter 
							new_row_data['event_duration'] = total_rest_duration
							new_df = pd.concat([new_df, pd.DataFrame([new_row_data])], ignore_index=True)
						start_counter += max_length_samples
						total_rest_duration -= max_length_samples
				else:
					total_rest_duration = 0
	if last_sample_n:
		start_counter = df.iloc[-1]['event_start'] + df.iloc[-1]['event_duration']
		total_rest_duration = last_sample_n - start_counter
		while total_rest_duration > 0:
			if total_rest_duration > min_rest_length_samples:
				new_row_data = {} 
				for col in df.columns:
					new_row_data[col] = rest_token
				if total_rest_duration > max_length_samples:
					new_row_data['event_start'] = start_counter 
					new_row_data['event_duration'] = max_length_samples
					new_df = pd.concat([new_df, pd.DataFrame([new_row_data])], ignore_index=True)
					start_counter += max_length_samples
					total_rest_duration -= max_length_samples
				else:
					# last rest
					if total_rest_duration > min_rest_length_samples:
						new_row_data['event_start'] = start_counter 
						new_row_data['event_duration'] = total_rest_duration
						new_df = pd.concat([new_df, pd.DataFrame([new_row_data])], ignore_index=True)
					start_counter += max_length_samples
					total_rest_duration -= max_length_samples
			else:
				total_rest_duration = 0
	new_df = pd.concat([new_df, pd.DataFrame([df.iloc[-1]])], ignore_index=True)
	return new_df


def extractFeatures(signal, 
					sample_rate, 
					FFT_window_size, 
					hop_size,
					silence_threshold=0.001,
					mfcc_N=13,
					chroma_N=12,
					compute_rests=True,
					max_event_duration_samples=44100, 
					min_event_duration_samples=441, 
					includeStd=True):

	allfeatures = ['rms', 'pitch', 'cent', 'flatness', 'rolloff', 'MFCC', 'chroma']
	feature_names, feature_names_continuous = getFeatureNames(allfeatures, mfcc_N=13, chroma_N=12, includeStd=includeStd)
	onsetDetector = OnsetDetection(sr=sample_rate)
	onsets = onsetDetector(signal.reshape(1,-1))
	newonsets = []
	latest_onset = 0
	for i in range(1,len(onsets)):
		if int(onsets[i]) - latest_onset > min_event_duration_samples:
			newonsets.append(onsets[i])
			latest_onset = onsets[i]
	onsets = newonsets
	print(f'detected {len(onsets)} onsets...')
	cont_features = computeFeatures(signal, sample_rate, FFT_window_size, hop_size, mfcc_N=mfcc_N, chroma_N=chroma_N)
	cont_features_df = pd.DataFrame(data=cont_features.T, columns=feature_names_continuous)
	featurestot = []
	rmstot = []
	centtot = []
	for i in range(len(onsets)-1):
		this_onset_signal = signal[onsets[i]:onsets[i+1]]
		features = computeFeaturesOnset(this_onset_signal, 
										sample_rate, 
										FFT_window_size, 
										hop_size, 
										max_length_samples=max_event_duration_samples,
										mfcc_N=mfcc_N, 
										chroma_N=chroma_N,
										silence_threshold=silence_threshold, 
										includeStd=includeStd)
		featurestot.append(features)
	if np.array(featurestot).shape[0] > 0:
		features_df = pd.DataFrame(data=np.array(featurestot), columns=feature_names)
		features_df['event_start'] = onsets[:-1]
		if 'event_duration' in features_df.columns and compute_rests:
			features_df = computeRests(features_df, 
										rest_token=0, 
										max_length_samples=max_event_duration_samples, 
										min_rest_length_samples=min_event_duration_samples, 
										last_sample_n=(signal.shape[0]-1))

		return features_df, cont_features_df
	else: 
		return None, cont_features_df


def computeCorpus(corpus_files, 
					save_path, 
					sample_rate, 
					FFT_window_size, 
					hop_size, 
					silence_threshold=0.001,
					plot_feature='rms_mean',
					compute_rests=True,
					max_event_duration_samples=44100,
					min_event_duration_samples=441,
					mfcc_N=13, 
					chroma_N=12):

	print(f'Loading audio corpus with {len(corpus_files)} files...')
	signals = []
	signal_onsets = []
	signals_dfs = []
	signals_dfs_cont = []	
	for audiofile in corpus_files:
		signal, _ = librosa.load(audiofile, sr=sample_rate, mono=True)
		rms = librosa.feature.rms(y=signal, frame_length=FFT_window_size, hop_length=hop_size)
		if rms.max() < 0.7: # normalize amplitude
			signal *= 0.6/signal.max()
		print(f'Loading audiofile {audiofile} with length {signal.shape[0]} samples')
		features_df, cont_features_df = extractFeatures(signal, 
														sample_rate, 
														FFT_window_size, 
														hop_size, 
														silence_threshold=silence_threshold,
														mfcc_N=mfcc_N, 
														chroma_N=chroma_N,
														compute_rests=compute_rests,
														max_event_duration_samples=max_event_duration_samples, 
														min_event_duration_samples=min_event_duration_samples, 
														includeStd=True)
		if features_df is not None:
			features_df['filename'] = [audiofile for _ in range(features_df.shape[0])]
			signals_dfs.append(features_df)
			cont_features_df['filename'] = [audiofile for _ in range(cont_features_df.shape[0])]
			signals_dfs_cont.append(cont_features_df)

	corpus_df = pd.concat(signals_dfs, axis=0, ignore_index=True)
	corpus_df.to_csv(f'{save_path}/corpus_discrete.csv')
	fig_save_dir = f'{save_path}/00_figures'
	os.makedirs(fig_save_dir, exist_ok=True)
	unique_filenames = corpus_df['filename'].unique()
	for filename in unique_filenames:
		song_name = filename.split('/')[-2].split('.')[0]
		dataVisualize.plotFromDB(save_path, 
								filename,
								plot_feature=plot_feature, 
								sample_start_s=0,
								save_dir=f'{fig_save_dir}/{song_name}_full')
		dataVisualize.plotFromDB(save_path, 
								filename,
								plot_feature=plot_feature, 
								sample_start_s=20,
								sample_duration_s=30,
								save_dir=f'{fig_save_dir}/{song_name}_short')
	corpus_df_cont = pd.concat(signals_dfs_cont, axis=0, ignore_index=True)
	corpus_df_cont.to_csv(f'{save_path}/corpus_continuous.csv')
	return corpus_df


if __name__ == "__main__":

	processing_params = {}
	processing_params['sample_rate'] = 44100
	processing_params['FFT_window_size'] = 2048
	processing_params['hop_size'] = 1024
	processing_params['num_MFCC'] = 13
	processing_params['num_chroma'] = 12
	processing_params['silence_threshold'] = 0.001
	processing_params['max_event_duration_samples'] = 88200
	processing_params['min_event_duration_samples'] = 882

	# extract 
	corpus_path = '00_corpus/guitar-duo'
	src_track_name = 'guitar1'
	tgt_track_name = 'guitar2'
	processing_params['track_name'] = src_track_name
	corpus_files = os.listdir(corpus_path)
	corpus_files = [f'{corpus_path}/{filename}/{processing_params['track_name']}.wav' for filename in corpus_files if os.path.isdir(f'{corpus_path}/{filename}') and filename != '00_process_src' and filename != '00_process_tgt']
	save_dir = f'{corpus_path}/00_process_src'
	os.makedirs(save_dir, exist_ok=True)
	with open(f'{save_dir}/featureExtraction_params.json', 'w', encoding='utf-8') as f:
		json.dump(processing_params, f, ensure_ascii=False, indent=4)

	df = computeCorpus(corpus_files, 
						save_dir,
						processing_params['sample_rate'], 
						processing_params['FFT_window_size'], 
						processing_params['hop_size'],
						compute_rests=True,
						silence_threshold=processing_params['silence_threshold'],
						max_event_duration_samples=processing_params['max_event_duration_samples'],
						min_event_duration_samples=processing_params['min_event_duration_samples'],
						mfcc_N=processing_params['num_MFCC'], 
						chroma_N=processing_params['num_chroma'])


	# USE SAME PARAMETERS FOR COUPLED TRACKS DATASET
	processing_params['track_name'] = tgt_track_name

	# corpus_path = '00_corpus/musdb-reverse'
	corpus_files = os.listdir(corpus_path)
	corpus_files = [f'{corpus_path}/{filename}/{processing_params['track_name']}.wav' for filename in corpus_files if os.path.isdir(f'{corpus_path}/{filename}') and filename != '00_process_src' and filename != '00_process_tgt']
	save_dir = f'{corpus_path}/00_process_tgt'
	os.makedirs(save_dir, exist_ok=True)
	with open(f'{save_dir}/featureExtraction_params.json', 'w', encoding='utf-8') as f:
		json.dump(processing_params, f, ensure_ascii=False, indent=4)

	df = computeCorpus(corpus_files, 
						save_dir,
						processing_params['sample_rate'], 
						processing_params['FFT_window_size'], 
						processing_params['hop_size'],
						compute_rests=True,
						silence_threshold=processing_params['silence_threshold'],
						max_event_duration_samples=processing_params['max_event_duration_samples'],
						min_event_duration_samples=processing_params['min_event_duration_samples'],
						mfcc_N=processing_params['num_MFCC'], 
						chroma_N=processing_params['num_chroma'])


	# plot all couples and save fig in their own folder
	# corpus_path = '00_corpus/musdb-reverse'
	corpus_dirs = os.listdir(corpus_path)
	corpus_dirs = [filename for filename in corpus_dirs if os.path.isdir(f'{corpus_path}/{filename}') and filename != '00_process_src' and filename != '00_process_tgt']
	for song in corpus_dirs:
		dataVisualize.plotCoupleFromDB(f'{corpus_path}/00_process_src/corpus_discrete.csv', 
									f'{corpus_path}/00_process_tgt/corpus_discrete.csv', 
									f'{corpus_path}/{song}/{src_track_name}.wav',
									f'{corpus_path}/{song}/{tgt_track_name}.wav',
									sample_rate=processing_params['sample_rate'],
									plot_feature_src='rms_mean', 
									plot_feature_tgt='rms_mean', 
									sample_start_s=0, 
									sample_duration_s=None,
									save_dir=f'{corpus_path}/{song}/{src_track_name}_{tgt_track_name}')
		dataVisualize.plotCoupleFromDB(f'{corpus_path}/00_process_src/corpus_discrete.csv', 
									f'{corpus_path}/00_process_tgt/corpus_discrete.csv', 
									f'{corpus_path}/{song}/{src_track_name}.wav',
									f'{corpus_path}/{song}/{tgt_track_name}.wav',
									sample_rate=processing_params['sample_rate'],
									plot_feature_src='rms_mean', 
									plot_feature_tgt='rms_mean', 
									sample_start_s=20, 
									sample_duration_s=30,
									save_dir=f'{corpus_path}/{song}/{src_track_name}_{tgt_track_name}_short')


