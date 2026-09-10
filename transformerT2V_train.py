import json
import time
import os
import random
import joblib
import math
import librosa

import numpy as np
import pandas as pd

from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import dataProcess
from transformerT2V import TransformerDiscreteT2V
# from transformerT2V_evaluate import renderModelOutput


def batchify(data, batch_size=16):
	# return batches of dim (batch_size, seq_length, num_feats)
	batches = []
	for i in range(0, len(data)-batch_size, batch_size):
		this_batch_X = []
		this_batch_y = []
		this_batch_y_tgt = []
		for k in range(batch_size):
			this_batch_X.append(data[i+k][0])
			this_batch_y.append(data[i+k][1])
			this_batch_y_tgt.append(data[i+k][2])
		batches.append([np.array(this_batch_X),np.array(this_batch_y),np.array(this_batch_y_tgt)])
	print(f"{len(batches)} batches of size {batch_size}")
	return batches


# train and validation loops
def trainLoop(model, train_dataloader, loss_fn, opt, device, PAD_TOKEN):
	model.train()
	total_loss = 0
	for batch in train_dataloader:
		X = torch.tensor(batch[0]).type(torch.FloatTensor).to(device)
		y = torch.tensor(batch[1]).type(torch.FloatTensor).to(device)
		y_expected = torch.tensor(batch[2]).type(torch.FloatTensor).to(device).reshape(batch[2].shape[0],1,batch[2].shape[1])
		# generate pad mask before t2v (t2v is non-linear and padding has to be applied before)
		src_pad_mask = model.create_pad_mask(X, pad_token=PAD_TOKEN).to(device)
		tgt_pad_mask = model.create_pad_mask(y, pad_token=PAD_TOKEN).to(device)
		tgt_mask = model.get_tgt_mask(y.size(1)).to(device)
		pred = model(X, y, tgt_mask=tgt_mask, src_pad_mask=src_pad_mask[:,:,0], tgt_pad_mask=tgt_pad_mask[:,:,0])
		loss = loss_fn(pred, y_expected)
		opt.zero_grad()
		loss.backward()
		opt.step()
		total_loss += loss.detach().item()
	return total_loss / len(train_dataloader)

def validationLoop(model, val_dataloader, loss_fn, opt, device, PAD_TOKEN):
	model.eval()
	total_loss = 0
	with torch.no_grad():
		for batch in val_dataloader:
			X = torch.tensor(batch[0]).type(torch.FloatTensor).to(device)
			y = torch.tensor(batch[1]).type(torch.FloatTensor).to(device)
			y_expected = torch.tensor(batch[2]).type(torch.FloatTensor).to(device).reshape(batch[2].shape[0],1,batch[2].shape[1])
			# generate pad mask before t2v (t2v is non-linear and padding has to be applied before)
			src_pad_mask = model.create_pad_mask(X, pad_token=PAD_TOKEN).to(device)
			tgt_pad_mask = model.create_pad_mask(y, pad_token=PAD_TOKEN).to(device)
			tgt_mask = model.get_tgt_mask(y.size(1)).to(device)
			pred = model(X, y, tgt_mask=tgt_mask, src_pad_mask=src_pad_mask[:,:,0], tgt_pad_mask=tgt_pad_mask[:,:,0])
			loss = loss_fn(pred, y_expected)
			total_loss += loss.detach().item()
	return total_loss / len(val_dataloader)


def spectral_loss(y_true, y_pred):
	loss = 0
	# Multiple FFT window sizes
	for n_fft in [512, 1024, 2048]:
		# Spectral convergence loss
		stft_true = torch.stft(y_true, n_fft=n_fft, return_complex=True)
		stft_pred = torch.stft(y_pred, n_fft=n_fft, return_complex=True)
		sc_loss = torch.norm(stft_true - stft_pred, 'fro') / torch.norm(stft_true, 'fro')
		# Log magnitude loss
		mag_true = torch.abs(stft_true)
		mag_pred = torch.abs(stft_pred)
		logmag_loss = F.l1_loss(torch.log(mag_true + 1e-7), torch.log(mag_pred + 1e-7))
		loss += sc_loss + logmag_loss
	return loss


def evaluateMrSTFT(model, val_dataloader, device, PAD_TOKEN, 
					tgt_scaler, target_corpus_df, features_target, 
					concatenative_corpus):
	model.eval()
	total_mrSTFT = []
	total_r2 = []
	with torch.no_grad():
		for batch in val_dataloader:
			X = torch.tensor(batch[0]).type(torch.FloatTensor).to(device)
			y = torch.tensor(batch[1]).type(torch.FloatTensor).to(device)
			# generate pad mask before t2v (t2v is non-linear and padding has to be applied before)
			src_pad_mask = model.create_pad_mask(X, pad_token=PAD_TOKEN).to(device)
			tgt_pad_mask = model.create_pad_mask(y, pad_token=PAD_TOKEN).to(device)
			tgt_mask = model.get_tgt_mask(y.size(1)).to(device)
			pred = model(X, y, tgt_mask=tgt_mask, src_pad_mask=src_pad_mask[:,:,0], tgt_pad_mask=tgt_pad_mask[:,:,0]).detach().cpu().numpy()
			# compute rMSE
			pred_denorm = tgt_scaler.inverse_transform(pred.reshape(pred.shape[0], pred.shape[2]))
			y_expected = batch[2]
			y_expected_denorm = tgt_scaler.inverse_transform(y_expected)
			r2 = r2_score(y_expected_denorm, pred_denorm)
			total_r2.append(r2)
			next_token = pred_denorm[-1,:]
			for i in range(y_expected_denorm.shape[0]):
				this_y_expected = y_expected_denorm[i,:]
				next_token = pred_denorm[i,:]
				pred_closest_index = (target_corpus_df[features_target] - next_token).abs().idxmin()
				y_expected_index = (target_corpus_df[features_target] - this_y_expected).abs().idxmin()
				# find sound segment
				closest_row = target_corpus_df.loc[pred_closest_index[0]]
				pred_sound_segment = concatenative_corpus[closest_row['filename']][int(closest_row['event_start']):int(closest_row['event_start'])+int(closest_row['event_duration'])]
				closest_row = target_corpus_df.loc[y_expected_index[0]]
				expected_sound_segment = concatenative_corpus[closest_row['filename']][int(closest_row['event_start']):int(closest_row['event_start'])+int(closest_row['event_duration'])]
				min_len = min(expected_sound_segment.shape[0], pred_sound_segment.shape[0])
				mrSTFT = spectral_loss(torch.tensor(expected_sound_segment[:min_len]), torch.tensor(pred_sound_segment[:min_len]))
				total_mrSTFT.append(mrSTFT.detach().item())
				print(f'{min_len}: {mrSTFT.detach().item():.3f}')
	#         total_mrSTFT += mrSTFT.detach().item() if mrSTFT.detach().item() != np.inf and mrSTFT.detach().item() != np.nan else 0
	# return total_mrSTFT / len(val_dataloader)
	total_mrSTFT = np.array(total_mrSTFT)
	mask = np.isfinite(total_mrSTFT)
	total_mrSTFT = total_mrSTFT[mask]
	return total_mrSTFT.mean(), np.array(total_r2).mean()


def evaluateAccuracyWithClustering(model, val_dataloader, device, PAD_TOKEN, 
									tgt_scaler, target_corpus_df, features_target, 
									tgt_brc):
	model.eval()
	total_accuracy = 0
	total_f1 = 0
	with torch.no_grad():
		for batch in val_dataloader:
			X = torch.tensor(batch[0]).type(torch.FloatTensor).to(device)
			y = torch.tensor(batch[1]).type(torch.FloatTensor).to(device)
			src_pad_mask = model.create_pad_mask(X, pad_token=PAD_TOKEN).to(device)
			tgt_pad_mask = model.create_pad_mask(y, pad_token=PAD_TOKEN).to(device)
			tgt_mask = model.get_tgt_mask(y.size(1)).to(device)
			pred = model(X, y, tgt_mask=tgt_mask, src_pad_mask=src_pad_mask[:,:,0], tgt_pad_mask=tgt_pad_mask[:,:,0]).detach().cpu().numpy()
			y_expected = batch[2]
			pred_labels = tgt_brc.predict(pred.reshape(pred.shape[0], pred.shape[2]))
			expected_labels = tgt_brc.predict(y_expected.reshape(pred.shape[0], pred.shape[2]))
			accuracy = sum(pred_labels == expected_labels) / expected_labels.shape[0]
			f1 = f1_score(expected_labels, pred_labels, average='weighted')
			total_accuracy += accuracy
			total_f1 += f1
	return total_accuracy / len(val_dataloader), total_f1 / len(val_dataloader)


def trainTransformerDiscreteT2V(training_parameters, save_dir, model_load_path=None):

	# MAKE SAVE DIRECTORY
	today = date.today()
	model_name = f'{int(time.time())}-transformerDiscreteT2V'
	logdir = f'{save_dir}/{str(today)}/{model_name}' # MAKE LOGDIR BY DAY
	os.makedirs(logdir, exist_ok=True)
	with open(f'{logdir}/training_config.json', 'w', encoding='utf-8') as f:
		json.dump(training_parameters, f, ensure_ascii=False, indent=4)

	# SAVE PROCESSING PARAMETERS
	with open(f'{training_parameters["source_corpus_path"]}/featureExtraction_params.json', 'r') as f:
		src_processing_params = json.load(f)
	with open(f'{training_parameters["target_corpus_path"]}/featureExtraction_params.json', 'r') as f:
		tgt_processing_params = json.load(f)

	# MAKE SURE DATASETS WERE COMPUTED WITH SAME PARAMETERS
	assert src_processing_params['sample_rate'] == tgt_processing_params['sample_rate']
	assert src_processing_params['FFT_window_size'] == tgt_processing_params['FFT_window_size']
	assert src_processing_params['hop_size'] == tgt_processing_params['hop_size']
	assert src_processing_params['num_MFCC'] == tgt_processing_params['num_MFCC']
	assert src_processing_params['num_chroma'] == tgt_processing_params['num_chroma']

	# get feat names from inputs
	features_source, _ = dataProcess.getFeatureNames(training_parameters['features_source'], 
													mfcc_N=src_processing_params['num_MFCC'], 
													chroma_N=src_processing_params['num_chroma'],
													includeStd=training_parameters['includeStd_src'])
	features_target, _ = dataProcess.getFeatureNames(training_parameters['features_target'], 
													mfcc_N=src_processing_params['num_MFCC'], 
													chroma_N=src_processing_params['num_chroma'],
													includeStd=training_parameters['includeStd_tgt'])


	# SAVE FEATURE EXTRACTION PARAMS
	src_processing_params['source_track_name'] = src_processing_params['track_name']
	src_processing_params['target_track_name'] = tgt_processing_params['track_name']
	del src_processing_params['track_name']
	with open(f'{logdir}/featureExtraction_params.json', 'w', encoding='utf-8') as f:
		json.dump(src_processing_params, f, ensure_ascii=False, indent=4)

	# SET SEED
	torch.manual_seed(training_parameters['seed'])
	random.seed(training_parameters['seed'])
	np.random.seed(training_parameters['seed'])


	# LOAD CORPUS
	source_corpus_df = pd.read_csv(f'{training_parameters["source_corpus_path"]}/corpus_discrete.csv', index_col=0)
	target_corpus_df = pd.read_csv(f'{training_parameters["target_corpus_path"]}/corpus_discrete.csv', index_col=0)
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
		partial_src_df = source_corpus_df[source_corpus_df['filename'] == f'{training_parameters["sound_corpus_path"]}/{filename}/{src_processing_params["source_track_name"]}.wav']
		partial_tgt_df = target_corpus_df[target_corpus_df['filename'] == f'{training_parameters["sound_corpus_path"]}/{filename}/{src_processing_params["target_track_name"]}.wav']
		# check that both are non-empty
		if partial_src_df.shape[0] > 0 and partial_tgt_df.shape[0] > 0:
			src_signals_dfs.append(partial_src_df)
			tgt_signals_dfs.append(partial_tgt_df)
		# src_signals_dfs.append(source_corpus_df[source_corpus_df['filename'] == f'{training_parameters['sound_corpus_path']}/{filename}/{src_processing_params['source_track_name']}.wav'])
		# tgt_signals_dfs.append(target_corpus_df[target_corpus_df['filename'] == f'{training_parameters['sound_corpus_path']}/{filename}/{src_processing_params['target_track_name']}.wav'])

	print(source_corpus_df.head())
	# print(target_corpus_df.head())
	# print(source_corpus_df.columns)

	# normalize source features
	src_features = source_corpus_df[features_source].values
	src_scaler = StandardScaler()
	src_scaler.fit(src_features)
	src_normalized_sequences = []
	for seq_df in src_signals_dfs:
		sequence_df = pd.DataFrame(data=src_scaler.transform(seq_df[features_source].values), columns=features_source)
		sequence_df['event_start'] = seq_df['event_start'].values
		src_normalized_sequences.append(sequence_df)
	joblib.dump(src_scaler, f'{logdir}/source_scaler.pkl')

	# normalize target features
	tgt_features = target_corpus_df[features_target].values
	tgt_scaler = StandardScaler()
	tgt_scaler.fit(tgt_features)
	tgt_normalized_sequences = []
	for seq_df in tgt_signals_dfs:
		sequence_df = pd.DataFrame(data=tgt_scaler.transform(seq_df[features_target].values), columns=features_target)
		sequence_df['event_start'] = seq_df['event_start'].values
		tgt_normalized_sequences.append(sequence_df)
	joblib.dump(tgt_scaler, f'{logdir}/target_scaler.pkl')


	# make dataloader
	window_size_s = training_parameters['window_size_s']
	pred_time_s = training_parameters['pred_time_s']
	WINDOW_N_SAMPLES = window_size_s * src_processing_params['sample_rate']
	PRED_TIME_SAMPLES = pred_time_s * src_processing_params['sample_rate']
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

	# DATA AUGMENTATION
	rmsTranspose = True
	pitchTranspose = True
	durationTranspose = True
	latentsTranspose = False
	maskAugment = False
	randomSubstitute = True
	NUM_AUGMENT = training_parameters['augment_amt']
	rmsIndexSrc = features_source.index("rms_mean")
	rmsIndexTgt = features_target.index("rms_mean")
	pitchIndexSrc = features_source.index("pitch_mean")
	pitchIndexTgt = features_target.index("pitch_mean")
	durationIndexSrc = features_source.index("event_duration")
	durationIndexTgt = features_target.index("event_duration")
	# latentIndexSrc = [features_source.index(f"latent-{j}") for j in range(training_parameters['N_latents_src'])]
	# latentIndexTgt = [features_source.index(f"latent-{j}") for j in range(training_parameters['N_latents_tgt'])]
	new_data = []
	for datum in data:
		if datum[0].shape[0] > 0 and datum[1].shape[0] > 0:
			x_src = datum[0].copy()
			x_tgt = datum[1].copy()
			y_tgt = datum[2].copy()
			new_data.append([x_src, x_tgt, y_tgt])
			for _ in range(NUM_AUGMENT):
				if rmsTranspose:
					# increase or decrease rms by max 10 %
					RMS_TRANSP = 0.2
					coefficient = RMS_TRANSP * (np.random.rand() * 2 -1)
					x_src[:,rmsIndexSrc] = x_src[:,rmsIndexSrc] + x_src[:,rmsIndexSrc] * coefficient
					x_tgt[:,rmsIndexTgt] = x_tgt[:,rmsIndexTgt] + x_tgt[:,rmsIndexTgt] * coefficient
					y_tgt[rmsIndexTgt] = y_tgt[rmsIndexTgt] + y_tgt[rmsIndexTgt] * coefficient
					# new_data.append([x_src, x_tgt, y_tgt])
				if pitchTranspose:
					# transpose pitch by max 10 %
					PITCH_TRANSP = 0.3
					coefficient = PITCH_TRANSP * (np.random.rand() * 2 -1)
					x_src[:,pitchIndexSrc] = x_src[:,pitchIndexSrc] + x_src[:,pitchIndexSrc] * coefficient
					x_tgt[:,pitchIndexTgt] = x_tgt[:,pitchIndexTgt] + x_tgt[:,pitchIndexTgt] * coefficient
					y_tgt[pitchIndexTgt] = y_tgt[pitchIndexTgt] + y_tgt[pitchIndexTgt] * coefficient
					# new_data.append([x_src, x_tgt, datum[2]])
				if durationTranspose:
					# increase or decrease durations by max 10 %
					DUR_TRANSP = 0.3
					coefficient = DUR_TRANSP * (np.random.rand() * 2 -1)
					x_src[:,durationIndexSrc] = x_src[:,durationIndexSrc] + x_src[:,durationIndexSrc] * coefficient
					x_tgt[:,durationIndexTgt] = x_tgt[:,durationIndexTgt] + x_tgt[:,durationIndexTgt] * coefficient
					y_tgt[durationIndexTgt] = y_tgt[durationIndexTgt] + y_tgt[durationIndexTgt] * coefficient
				# if latentsTranspose:
				# 	# increase or decrease durations by max 10 %
				# 	LATENTS_TRANSP = 0.2
				# 	coefficient = LATENTS_TRANSP * (np.random.rand(len(latentIndexSrc)) * 2 -1)
				# 	x_src[:,latentIndexSrc] = x_src[:,latentIndexSrc] + x_src[:,latentIndexSrc] * coefficient
				# 	coefficient = LATENTS_TRANSP * (np.random.rand(len(latentIndexTgt)) * 2 -1)
				# 	x_tgt[:,latentIndexTgt] = x_tgt[:,latentIndexTgt] + x_tgt[:,latentIndexTgt] * coefficient
				# 	y_tgt[latentIndexTgt] = y_tgt[latentIndexTgt] + y_tgt[latentIndexTgt] * coefficient
				if maskAugment:
					MASK_TOKEN = np.ones_like(x_src[0,:-1]) * -10
					src_len, tgt_len = x_src.shape[0], x_tgt.shape[0]
					if np.random.rand() > 0.8:
						x_src[random.randint(0, src_len-1), :-1] = MASK_TOKEN
						x_tgt[random.randint(0, tgt_len-1), :-1] = MASK_TOKEN
				if randomSubstitute:
					src_len, tgt_len = x_src.shape[0]-1, x_tgt.shape[0]-1
					if np.random.rand() > 0.8:
						x_src[random.randint(0, src_len), :-1] = x_src[random.randint(0, src_len), :-1]
						x_tgt[random.randint(0, tgt_len), :-1] = x_tgt[random.randint(0, tgt_len), :-1]
				new_data.append([x_src, x_tgt, y_tgt])
	non_augmented_data = data.copy()
	data = new_data


	MAX_SEQ_LEN = int(np.array(seq_lens).mean() + np.array(seq_lens).std())
	if model_load_path:
		MAX_SEQ_LEN = training_parameters['max_sequence_length']
	# remove rows where time is less than min or 
	# normalize time feature
	src_time_scaler = StandardScaler()
	src_time_scaler.fit(np.array(all_src_times).reshape(-1,1))
	tgt_time_scaler = StandardScaler()
	tgt_time_scaler.fit(np.array(all_tgt_times).reshape(-1,1))
	joblib.dump(src_time_scaler, f'{logdir}/source_time_scaler.pkl')
	joblib.dump(tgt_time_scaler, f'{logdir}/target_time_scaler.pkl')
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

	with open(f'{logdir}/training_config.json', 'r') as f:
		training_log = json.load(f)
	training_log['max_sequence_length'] = MAX_SEQ_LEN
	with open(f'{logdir}/training_config.json', 'w', encoding='utf-8') as f:
		json.dump(training_log, f, ensure_ascii=False, indent=4)

	# shuffle sequences
	np.random.shuffle(data)

	# train-validation split
	train_perc = training_parameters['dataset_training_percentage']
	val_perc = training_parameters['dataset_validation_percentage']
	train_data = data[:int(len(data)*train_perc)]
	test_data = data[int(len(data)*train_perc):]
	val_data = train_data[int(len(train_data)*(1-val_perc)):]
	train_data = train_data[:int(len(train_data)*(1-val_perc))]

	# val_data = data[int(len(data)*train_perc):int(len(data)*(train_perc+val_perc))]
	# test_data = data[int(len(data)*(train_perc+val_perc)):]

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
			max_seq_len=MAX_SEQ_LEN,
			dim_model=training_parameters['dim_model'], 
			num_heads=training_parameters['num_heads'], 
			num_encoder_layers=training_parameters['num_encoder_layers'], 
			num_decoder_layers=training_parameters['num_decoder_layers'], 
			dim_feedforward=training_parameters['dim_feedforward'], 
			dropout_p=training_parameters['dropout_p']
	).to(device)
	if model_load_path:
		model.load_state_dict(torch.load(model_load_path, weights_only=True, map_location=device))
	opt = torch.optim.SGD(model.parameters(), lr=training_parameters['learning_rate'])
	# loss_fn = nn.CrossEntropyLoss()
	if training_parameters['loss_fn'] == 'L1':
		loss_fn = nn.L1Loss()
	else:
		loss_fn = nn.MSELoss()


	# Used for plotting later on
	train_loss_list, validation_loss_list = [], []

	writer = SummaryWriter(logdir)
	epochs = training_parameters['epochs']
	best_loss = float('inf')
	print("Training and validating model")
	for epoch in range(epochs):
		print("-"*25, f"Epoch {epoch + 1}","-"*25)

		train_loss = trainLoop(model, train_dataloader, loss_fn, opt, device, training_parameters['pad_token'])
		train_loss_list += [train_loss]
		writer.add_scalar("Loss/train", train_loss, epoch)

		validation_loss = validationLoop(model, val_dataloader, loss_fn, opt, device, training_parameters['pad_token'])
		validation_loss_list += [validation_loss]
		writer.add_scalar("Loss/validation", validation_loss, epoch)

		print(f"Training loss: {train_loss:.4f}")
		print(f"Validation loss: {validation_loss:.4f}")
		print()

		if validation_loss < best_loss:
			best_loss = validation_loss
			torch.save(model.state_dict(), f'{logdir}/best_model.pth')

	writer.flush()
	writer.close()


	evaluation_metrics = {}

	# load best model weights
	model_load_path = f'{logdir}/best_model.pth'
	model.load_state_dict(torch.load(model_load_path, weights_only=True))
	train_loss = validationLoop(model, train_dataloader, loss_fn, opt, device, training_parameters['pad_token'])
	evaluation_metrics['training_MSE'] = train_loss
	validation_loss = validationLoop(model, val_dataloader, loss_fn, opt, device, training_parameters['pad_token'])
	evaluation_metrics['validation_MSE'] = validation_loss
	test_loss = validationLoop(model, test_dataloader, loss_fn, opt, device, training_parameters['pad_token'])
	evaluation_metrics['test_MSE'] = test_loss


	# LOAD TARGET CORPUS SOUNDFILES FOR CONCATENATIVE SYNTH
	target_corpus_df = pd.read_csv(f'{training_parameters["target_corpus_path"]}/corpus_discrete.csv', index_col=0)
	unique_filenames = unique_elements = list(set(target_corpus_df['filename'].values))
	concatenative_corpus = {}
	for audiofilepath in unique_filenames:
		signal, _ = librosa.load(audiofilepath, sr=src_processing_params['sample_rate'], mono=True)
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
		tgt_brc = joblib.load(f'{training_parameters["target_corpus_path"]}/tgt_birch_classifier.pkl')
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

	with open(f'{logdir}/evaluation.json', 'w', encoding='utf-8') as f:
		json.dump(evaluation_metrics, f, ensure_ascii=False, indent=4)


	# EVALUATE MODEL
	N_tracks = 4
	corpus_files = os.listdir(training_parameters['sound_corpus_path'])
	corpus_files = [filename for filename in corpus_files if os.path.isdir(f'{training_parameters["sound_corpus_path"]}/{filename}') and filename != '00_process_src' and filename != '00_process_tgt']
	eval_track_names = corpus_files[:4]
	# renderModelOutput(logdir, eval_track_names)

	print()
	print('FINISHED TRAINING')
	print('-'*20)
	print(f"training MSE: {evaluation_metrics['training_MSE']}")
	print(f"validation MSE: {evaluation_metrics['validation_MSE']}")
	print(f"test MSE: {evaluation_metrics['test_MSE']}")
	print(f"training R2: {evaluation_metrics['train_r2']}")
	print(f"validation R2: {evaluation_metrics['val_r2']}")
	print(f"test R2: {evaluation_metrics['test_r2']}")
	print(f"training mrSTFT: {evaluation_metrics['training_mrSTFT']}")
	print(f"validation mrSTFT: {evaluation_metrics['validation_mrSTFT']}")
	print(f"test mrSTFT: {evaluation_metrics['validation_mrSTFT']}")
	try:
		print(f"training accuracy: {evaluation_metrics['training accuracy']}")
		print(f"validation accuracy: {evaluation_metrics['validation accuracy']}")
		print(f"test accuracy: {evaluation_metrics['test accuracy']}")
		print(f"training F1: {evaluation_metrics['training F1']}")
		print(f"validation F1: {evaluation_metrics['validation F1']}")
		print(f"test F1: {evaluation_metrics['test F1']}")
	except:
		print("no accuracy score")
	print('-'*20)
	print()

	return logdir





if __name__ == "__main__":

	# SET TRAINING PARAMETERS
	training_parameters = {}
	training_parameters['source_corpus_path'] = '00_corpus/guitar-duo/00_process_src'
	training_parameters['target_corpus_path'] = '00_corpus/guitar-duo/00_process_tgt'
	training_parameters['sound_corpus_path'] = '00_corpus/guitar-duo'
	training_parameters['features_source'] = ['rms', 'chroma', 'pitch', 'event_duration']
	training_parameters['features_target'] = ['rms', 'chroma', 'pitch', 'event_duration']
	training_parameters['includeStd_src'] = False
	training_parameters['includeStd_tgt'] = False
	training_parameters['batch_size'] = 64
	training_parameters['num_frequency'] = 8
	training_parameters['dataset_training_percentage'] = 0.8
	training_parameters['dataset_validation_percentage'] = 0.1
	training_parameters['dim_model'] = 32
	training_parameters['num_heads'] = 8
	training_parameters['num_encoder_layers'] = 8
	training_parameters['num_decoder_layers'] = 8
	training_parameters['dim_feedforward'] = 512
	training_parameters['dropout_p'] = 0.3
	training_parameters['window_size_s'] = 8
	training_parameters['pred_time_s'] = 0
	training_parameters['pad_token'] = -1
	training_parameters['loss_fn'] = 'MSE'
	training_parameters['learning_rate'] = 0.001
	training_parameters['epochs'] = 2000
	training_parameters['seed'] = 666
	training_parameters['augment_amt'] = 20
	save_dir = '01_model_logs'

	# TRAIN MODEL
	model_logdir = trainTransformerDiscreteT2V(training_parameters, save_dir)

