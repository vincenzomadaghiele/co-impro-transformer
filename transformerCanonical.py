import os
import json
import time
import math
import random
import joblib
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import date
from sklearn.decomposition import PCA
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import dataProcess
from FO import FactorOracle
# from transformerDiscreteT2V_train import batchify


class VanillaTransformer(nn.Module):
    def __init__(self, 
                    num_tokens_src,
                    num_tokens_tgt,
                    dim_model, 
                    num_heads, 
                    num_encoder_layers, 
                    num_decoder_layers, 
                    dim_feedforward,
                    pad_idx,
                    dropout_p):
        super().__init__()
        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model
        # LAYERS
        self.src_embedding = nn.Embedding(num_tokens_src, dim_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(num_tokens_tgt, dim_model, padding_idx=pad_idx)
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, 
                                                    dropout_p=dropout_p, 
                                                    max_len=5000)
        self.transformer = nn.Transformer(d_model=dim_model,
                                        nhead=num_heads,
                                        num_encoder_layers=num_encoder_layers,
                                        num_decoder_layers=num_decoder_layers,
                                        dropout=dropout_p,
                                        dim_feedforward=dim_feedforward)
        self.out = nn.Linear(dim_model, num_tokens_tgt)
    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        src = src.reshape(src.shape[0], src.shape[1])
        tgt = tgt.reshape(tgt.shape[0], tgt.shape[1])
        src = self.src_embedding(src) * math.sqrt(self.dim_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)
        transformer_out = self.transformer(src, 
                                            tgt, 
                                            tgt_mask=tgt_mask, 
                                            src_key_padding_mask=src_pad_mask, 
                                            tgt_key_padding_mask=tgt_pad_mask)
        # transformer_out = self.transformer(src, tgt)
        # transformer_out = transformer_out[0,:,:].reshape(1, transformer_out.shape[1], transformer_out.shape[2])
        out = self.out(transformer_out).permute(1,0,2)
        return out#.permute(1,0,2)
    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        return mask
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        return (matrix == pad_token)


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


# train and validation loops
def trainLoop(model, train_dataloader, loss_fn, opt, device, PAD_TOKEN):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        X = torch.tensor(batch[0]).type(torch.int64).to(device)
        y = torch.tensor(batch[1]).type(torch.int64).to(device)
        y_expected = torch.tensor(batch[2]).type(torch.int64).to(device).reshape(batch[2].shape[0],1,batch[2].shape[1])
        # generate pad mask before t2v (t2v is non-linear and padding has to be applied before)
        src_pad_mask = model.create_pad_mask(X, pad_token=PAD_TOKEN).to(device)
        tgt_pad_mask = model.create_pad_mask(y, pad_token=PAD_TOKEN).to(device)        
        tgt_mask = model.get_tgt_mask(y.size(1)).to(device)
        pred = model(X, y, tgt_mask=tgt_mask, src_pad_mask=src_pad_mask[:,:,0], tgt_pad_mask=tgt_pad_mask[:,:,0])
        # print(pred[:,-1,:].shape)
        # print(y_expected.reshape(-1).shape)
        loss = loss_fn(pred[:,-1,:], y_expected.reshape(-1))
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
            X = torch.tensor(batch[0]).type(torch.int64).to(device)
            y = torch.tensor(batch[1]).type(torch.int64).to(device)
            y_expected = torch.tensor(batch[2]).type(torch.int64).to(device).reshape(batch[2].shape[0],1,batch[2].shape[1])
            # generate pad mask before t2v (t2v is non-linear and padding has to be applied before)
            src_pad_mask = model.create_pad_mask(X, pad_token=PAD_TOKEN).to(device)
            tgt_pad_mask = model.create_pad_mask(y, pad_token=PAD_TOKEN).to(device)
            tgt_mask = model.get_tgt_mask(y.size(1)).to(device)
            pred = model(X, y, tgt_mask=tgt_mask, src_pad_mask=src_pad_mask[:,:,0], tgt_pad_mask=tgt_pad_mask[:,:,0])
            loss = loss_fn(pred[:,-1,:], y_expected.reshape(-1))
            total_loss += loss.detach().item()
    return total_loss / len(val_dataloader)

def accuracy(model, val_dataloader, loss_fn, opt, device, PAD_TOKEN):
    model.eval()
    total_accuracy = 0
    total_f1 = 0
    tot_mrSTFT = []
    with torch.no_grad():
        for batch in val_dataloader:
            X = torch.tensor(batch[0]).type(torch.int64).to(device)
            y = torch.tensor(batch[1]).type(torch.int64).to(device)
            y_expected = torch.tensor(batch[2]).type(torch.int64).to(device).reshape(batch[2].shape[0],1,batch[2].shape[1])
            # generate pad mask before t2v (t2v is non-linear and padding has to be applied before)
            src_pad_mask = model.create_pad_mask(X, pad_token=PAD_TOKEN).to(device)
            tgt_pad_mask = model.create_pad_mask(y, pad_token=PAD_TOKEN).to(device)
            tgt_mask = model.get_tgt_mask(y.size(1)).to(device)
            pred = model(X, y, tgt_mask=tgt_mask, src_pad_mask=src_pad_mask[:,:,0], tgt_pad_mask=tgt_pad_mask[:,:,0])
            next_item = torch.topk(pred, k=1)[1]#.view(-1)[-1]#.item() # num with highest probability
            next_item_preds = next_item.detach().cpu().numpy().reshape(next_item.shape[0],next_item.shape[1])[:,0]
            accuracy = sum(next_item_preds == y_expected.detach().cpu().numpy().reshape(-1)) / next_item_preds.shape[0]
            f1 = f1_score(y_expected.detach().cpu().numpy().reshape(-1), next_item_preds, average='weighted')
            total_accuracy += accuracy
            total_f1 += f1

            mrSTFTs = []
            for i in range(next_item_preds.shape[0]):
                pred_sound_segment = random.choice(tgtAudioSamplesByClass[next_item_preds[i]])
                expected_sound_segment = batch[3][i]
                min_len = min(expected_sound_segment.shape[0], pred_sound_segment.shape[0])
                mrSTFT = spectral_loss(torch.tensor(expected_sound_segment[:min_len]), torch.tensor(pred_sound_segment[:min_len]))
                mrSTFTs.append(mrSTFT.detach().item() if mrSTFT.detach().item() != np.inf and mrSTFT.detach().item() != np.nan else 0)
            mrSTFTs = np.array(mrSTFTs)
            mask = np.isfinite(mrSTFTs)
            mrSTFTs = mrSTFTs[mask]
            tot_mrSTFT.append(np.array(mrSTFTs).mean())
    return total_accuracy / len(val_dataloader), total_f1 / len(val_dataloader), np.array(tot_mrSTFT).mean()


def batchify(data, batch_size=16):
    # return batches of dim (batch_size, seq_length, num_feats)
    batches = []
    for i in range(0, len(data)-batch_size, batch_size):
        this_batch_X = []
        this_batch_y = []
        this_batch_y_tgt = []
        this_batch_y_tgt_sound = []
        for k in range(batch_size):
            this_batch_X.append(data[i+k][0])
            this_batch_y.append(data[i+k][1])
            this_batch_y_tgt.append(data[i+k][2])
            this_batch_y_tgt_sound.append(data[i+k][3])
        batches.append([np.array(this_batch_X),np.array(this_batch_y),np.array(this_batch_y_tgt), this_batch_y_tgt_sound])
    print(f"{len(batches)} batches of size {batch_size}")
    return batches


def spectral_loss(y_true, y_pred):
    loss = 0
    # Multiple FFT window sizes
    for n_fft in [512, 1024, 2048]:
        # Spectral convergence loss
        if y_true.shape[0] > n_fft:
            stft_true = torch.stft(y_true, n_fft=n_fft, return_complex=True)
            stft_pred = torch.stft(y_pred, n_fft=n_fft, return_complex=True)
            sc_loss = torch.norm(stft_true - stft_pred, 'fro') / torch.norm(stft_true, 'fro')
            # Log magnitude loss
            mag_true = torch.abs(stft_true)
            mag_pred = torch.abs(stft_pred)
            logmag_loss = F.l1_loss(torch.log(mag_true + 1e-7), torch.log(mag_pred + 1e-7))
            loss += sc_loss + logmag_loss
    return loss


if __name__ == "__main__":


    databases = [
                'BassDrums', 
                'BassGuit', 
                'DrumsBass', 
                'DrumsGuit', 
                'GuitBass', 
                'GuitDrums'
                ]
    srcFeats = [
                ['rms', 'pitch', 'cent', 'event_duration'],
                ['rms', 'chroma', 'pitch', 'event_duration'],
                ['rms', 'flatness', 'cent', 'rolloff', 'event_duration'],
                ['rms', 'flatness', 'cent', 'rolloff', 'event_duration'],
                ['rms', 'chroma', 'pitch', 'event_duration'],
                ['rms', 'chroma', 'pitch', 'event_duration']
                ]
    tgtFeats = [
                ['rms', 'flatness', 'cent', 'rolloff', 'event_duration'],
                ['rms', 'chroma', 'pitch', 'event_duration'],
                ['rms', 'pitch', 'cent', 'event_duration'],
                ['rms', 'chroma', 'pitch', 'event_duration'],
                ['rms', 'chroma', 'pitch', 'event_duration'],
                ['rms', 'flatness', 'cent', 'rolloff', 'event_duration']
                ]

    for i in range(len(databases)):

        model_load_path = None

        # SET TRAINING PARAMETERS
        training_parameters = {}
        training_parameters['source_corpus_path'] = f'00_corpus/moisesdb/moisesdb-{databases[i]}/00_process_src'
        training_parameters['target_corpus_path'] = f'00_corpus/moisesdb/moisesdb-{databases[i]}/00_process_tgt'
        training_parameters['sound_corpus_path'] = f'00_corpus/moisesdb/moisesdb-{databases[i]}'
        training_parameters['features_source'] = srcFeats[i]
        training_parameters['features_target'] = tgtFeats[i]
        training_parameters['N_clusters_src'] = 50
        training_parameters['N_clusters_tgt'] = 50
        training_parameters['includeStd_src'] = False
        training_parameters['includeStd_tgt'] = False
        training_parameters['batch_size'] = 16
        training_parameters['dataset_training_percentage'] = 0.7
        training_parameters['dataset_validation_percentage'] = 0.2
        training_parameters['dim_model'] = 32
        training_parameters['num_heads'] = 8
        training_parameters['num_encoder_layers'] = 8
        training_parameters['num_decoder_layers'] = 8
        training_parameters['dim_feedforward'] = 32
        training_parameters['dropout_p'] = 0.3
        training_parameters['window_size_s'] = 6
        training_parameters['pred_time_s'] = 0
        training_parameters['pad_token'] = training_parameters['N_clusters_src']+1
        training_parameters['learning_rate'] = 0.001
        training_parameters['epochs'] = 1000
        training_parameters['seed'] = 666
        save_dir = '01_model_logs/PAPER_TESTS'


        # MAKE SAVE DIRECTORY
        today = date.today()
        model_name = f'{int(time.time())}-vanillaTransformer'
        logdir = f'{save_dir}/{str(today)}/{model_name}' # MAKE LOGDIR BY DAY
        os.makedirs(logdir, exist_ok=True)
        with open(f'{logdir}/training_config.json', 'w', encoding='utf-8') as f:
            json.dump(training_parameters, f, ensure_ascii=False, indent=4)

        # SAVE PROCESSING PARAMETERS
        with open(f'{training_parameters['source_corpus_path']}/featureExtraction_params.json', 'r') as f:
            src_processing_params = json.load(f)
        with open(f'{training_parameters['target_corpus_path']}/featureExtraction_params.json', 'r') as f:
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
        with open(f'{logdir}/featureExtraction_params.json', 'w', encoding='utf-8') as f:
            json.dump(src_processing_params, f, ensure_ascii=False, indent=4)

        # SET SEED
        torch.manual_seed(training_parameters['seed'])
        random.seed(training_parameters['seed'])
        np.random.seed(training_parameters['seed'])

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
            partial_src_df = source_corpus_df[source_corpus_df['filename'] == f'{training_parameters['sound_corpus_path']}/{filename}/{src_processing_params['source_track_name']}.wav']
            partial_tgt_df = target_corpus_df[target_corpus_df['filename'] == f'{training_parameters['sound_corpus_path']}/{filename}/{src_processing_params['target_track_name']}.wav']
            # check that both are non-empty
            if partial_src_df.shape[0] > 0 and partial_tgt_df.shape[0] > 0:
                src_signals_dfs.append(partial_src_df)
                tgt_signals_dfs.append(partial_tgt_df)

        print(source_corpus_df.head())
        # print(source_corpus_df.columns)


        # CLUSTER INTO CLASSES BASED ON FEATURES
        # normalize source features
        src_features = source_corpus_df[features_source].values
        src_scaler = StandardScaler()
        src_scaler.fit(src_features)
        src_normalized_sequences = []
        for seq_df in src_signals_dfs:
            sequence_df = pd.DataFrame(data=src_scaler.transform(seq_df[features_source].values), columns=features_source)
            sequence_df['event_start'] = seq_df['event_start'].values
            src_normalized_sequences.append(sequence_df)
        # joblib.dump(src_scaler, f'{logdir}/source_scaler.pkl')

        pca = PCA(n_components=2)
        X_scaled = src_scaler.transform(src_features)
        X_pca = pca.fit_transform(X_scaled)
        # if os.path.isfile(f'{training_parameters['source_corpus_path']}/src_birch_classifier.pkl'):
        #     brc = joblib.load(f'{training_parameters['source_corpus_path']}/src_birch_classifier.pkl')
        # else:
        brc = Birch(n_clusters=training_parameters['N_clusters_src']).fit(X_scaled)
        print("Explained variance:", pca.explained_variance_ratio_)
        # plt.scatter(X_pca[:,0], X_pca[:,1], c=brc.labels_, cmap='viridis')
        # plt.show()

        for seq_df in src_normalized_sequences:
            labels = brc.predict(seq_df.values[:,:-1])
            seq_df['labels'] = labels
            # FO = FactorOracle()
            # FO.train(labels)
            # # FO.visualize()
        joblib.dump(brc, f'{training_parameters['source_corpus_path']}/src_birch_classifier.pkl')


        # normalize target features
        tgt_features = target_corpus_df[features_target].values
        tgt_scaler = StandardScaler()
        tgt_scaler.fit(tgt_features)
        tgt_normalized_sequences = []
        for seq_df in tgt_signals_dfs:
            sequence_df = pd.DataFrame(data=tgt_scaler.transform(seq_df[features_target].values), columns=features_target)
            sequence_df['event_start'] = seq_df['event_start'].values
            tgt_normalized_sequences.append(sequence_df)
        # joblib.dump(tgt_scaler, f'{logdir}/target_scaler.pkl')


        pca = PCA(n_components=2)
        X_scaled = tgt_scaler.transform(tgt_features)
        X_pca = pca.fit_transform(X_scaled)
        # if os.path.isfile(f'{training_parameters['target_corpus_path']}/tgt_birch_classifier.pkl'):
        #     brc = joblib.load(f'{training_parameters['target_corpus_path']}/tgt_birch_classifier.pkl')
        # else:
        brc = Birch(n_clusters=training_parameters['N_clusters_tgt']).fit(X_scaled)
        print("Explained variance:", pca.explained_variance_ratio_)
        # plt.scatter(X_pca[:,0], X_pca[:,1], c=brc.labels_, cmap='viridis')
        # plt.show()

        for seq_df in tgt_normalized_sequences:
            labels = brc.predict(seq_df.values[:,:-1])
            seq_df['labels'] = labels
            # FO = FactorOracle()
            # FO.train(labels)
            # FO.visualize()
        joblib.dump(brc, f'{training_parameters['target_corpus_path']}/tgt_birch_classifier.pkl')


        # LOAD TARGET CORPUS SOUNDFILES FOR CONCATENATIVE SYNTH
        target_corpus_df = pd.read_csv(f'{training_parameters['target_corpus_path']}/corpus_discrete.csv', index_col=0)
        unique_filenames = unique_elements = list(set(target_corpus_df['filename'].values))
        concatenative_corpus = {}
        for audiofilepath in unique_filenames:
            signal, _ = librosa.load(audiofilepath, sr=src_processing_params['sample_rate'], mono=True)
            concatenative_corpus[audiofilepath] = signal
        tgtAudioSamplesByClass = {}
        for i in range(training_parameters['N_clusters_tgt']):
            tgtAudioSamplesByClass[i] = []


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
            X_values_feats = src_normalized_sequences[k][features_source].values
            y_values_feats = tgt_normalized_sequences[k][features_target].values
            X_values = src_normalized_sequences[k][['labels']].values
            y_values = tgt_normalized_sequences[k][['labels']].values
            X_times = src_normalized_sequences[k]['event_start'].values
            y_times = tgt_normalized_sequences[k]['event_start'].values
            for j in range(y_times.shape[0]-1):
                y_tgt = y_values[j+1,:]
                y_tgt_feats = y_values_feats[j+1,:]
                # we subtract inference time 
                # so that we can call inference some time before the end of an onset
                window_start_idx = (y_times[j+1] - PRED_TIME_SAMPLES) - WINDOW_N_SAMPLES
                indices = np.where((X_times >= window_start_idx) & (X_times < y_times[j+1]))
                # add time difference index
                x_src_window_times = (y_times[j+1] - PRED_TIME_SAMPLES) - X_times[indices]
                # x_src = np.concatenate((X_values[indices], x_src_window_times.reshape(-1,1)), axis=1)
                x_src = X_values[indices]
                all_src_times += x_src_window_times.tolist()
                indices = np.where((y_times >= window_start_idx) & (y_times < y_times[j+1]))
                # add time difference index
                x_tgt_window_times = (y_times[j+1] - PRED_TIME_SAMPLES) - y_times[indices]
                # x_tgt = np.concatenate((y_values[indices], x_tgt_window_times.reshape(-1,1)), axis=1)
                x_tgt = y_values[indices]
                all_tgt_times += x_tgt_window_times.tolist()

                tgt_duration = tgt_scaler.inverse_transform(y_tgt_feats.reshape(1,-1))[:,-1][0]
                y_tgt_audio = concatenative_corpus[f'{training_parameters['sound_corpus_path']}/{common_filenames[k]}/{src_processing_params['target_track_name']}.wav'][int(y_times[j+1]):int(y_times[j+1]+tgt_duration)]
                tgtAudioSamplesByClass[tgt_normalized_sequences[k][['labels']].values[j+1][0]].append(y_tgt_audio)

                data.append([x_src, x_tgt, y_tgt, y_tgt_audio])
                seq_lens.append(x_src.shape[0])
                seq_lens.append(x_tgt.shape[0])


        MAX_SEQ_LEN = int(np.array(seq_lens).mean() + np.array(seq_lens).std())
        if model_load_path is not None:
            MAX_SEQ_LEN = training_parameters['max_sequence_length']
        new_data = []
        for datum in data:
            x_src = datum[0][-MAX_SEQ_LEN:,:]
            x_tgt = datum[1][-MAX_SEQ_LEN:,:]
            new_data.append([x_src, x_tgt, datum[2], datum[3]])
        data = new_data


        # PAD SEQUENCES
        PAD_TOKEN = training_parameters['pad_token']
        new_data = []
        for datum in data:
            x_src = np.ones((MAX_SEQ_LEN, 1)) * PAD_TOKEN
            x_tgt = np.ones((MAX_SEQ_LEN, 1)) * PAD_TOKEN
            x_src[:datum[0].shape[0],:] = datum[0]
            x_tgt[:datum[1].shape[0],:] = datum[1]
            new_data.append([x_src, x_tgt, datum[2], datum[3]])
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
        val_data = data[int(len(data)*train_perc):int(len(data)*(train_perc+val_perc))]
        test_data = data[int(len(data)*(train_perc+val_perc)):]

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
        model = VanillaTransformer(
                num_tokens_src=training_parameters['N_clusters_src']+2, 
                num_tokens_tgt=training_parameters['N_clusters_tgt']+2, 
                dim_model=training_parameters['dim_model'], 
                num_heads=training_parameters['num_heads'], 
                num_encoder_layers=training_parameters['num_encoder_layers'], 
                num_decoder_layers=training_parameters['num_decoder_layers'], 
                dim_feedforward=training_parameters['dim_feedforward'], 
                pad_idx=training_parameters['pad_token'], 
                dropout_p=training_parameters['dropout_p']
        ).to(device)
        if model_load_path:
            model.load_state_dict(torch.load(model_load_path, weights_only=True, map_location=device))
        opt = torch.optim.SGD(model.parameters(), lr=training_parameters['learning_rate'])
        loss_fn = nn.CrossEntropyLoss()


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


        # logdir = '01_model_logs/tests/2026-01-15/1768479616-vanillaTransformer'
        evaluation_metrics = {}

        # load best model weights
        model_load_path = f'{logdir}/best_model.pth'
        model.load_state_dict(torch.load(model_load_path, weights_only=True))
        train_loss = validationLoop(model, train_dataloader, loss_fn, opt, device, training_parameters['pad_token'])
        evaluation_metrics['training_loss'] = train_loss
        validation_loss = validationLoop(model, val_dataloader, loss_fn, opt, device, training_parameters['pad_token'])
        evaluation_metrics['validation_loss'] = validation_loss
        test_loss = validationLoop(model, test_dataloader, loss_fn, opt, device, training_parameters['pad_token'])
        evaluation_metrics['test_loss'] = test_loss

        train_accuracy, train_f1, train_mrSTFT = accuracy(model, train_dataloader, loss_fn, 
                                        opt, device, training_parameters['pad_token'])
        val_accuracy, val_f1, val_mrSTFT = accuracy(model, val_dataloader, loss_fn, 
                                        opt, device, training_parameters['pad_token'])
        test_accuracy, test_f1, test_mrSTFT = accuracy(model, test_dataloader, loss_fn, 
                                        opt, device, training_parameters['pad_token'])

        evaluation_metrics['train_accuracy'] = train_accuracy
        evaluation_metrics['validation_accuracy'] = val_accuracy
        evaluation_metrics['test_accuracy'] = test_accuracy
        evaluation_metrics['train_f1'] = train_f1
        evaluation_metrics['val_f1'] = val_f1
        evaluation_metrics['test_f1'] = test_f1
        evaluation_metrics['train_mrSTFT'] = train_mrSTFT
        evaluation_metrics['val_mrSTFT'] = val_mrSTFT
        evaluation_metrics['test_mrSTFT'] = test_mrSTFT

        with open(f'{logdir}/evaluation.json', 'w', encoding='utf-8') as f:
            json.dump(evaluation_metrics, f, ensure_ascii=False, indent=4)







