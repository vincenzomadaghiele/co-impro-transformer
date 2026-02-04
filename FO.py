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
import matplotlib.patches as patches

from datetime import date
from sklearn.decomposition import PCA
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F

import dataProcess


class FactorOracle:
    def __init__(self):
        self.symbols = [] # symbols
        self.sigma = [] # for each state, stores a list of transitions for each symbol
        self.S = [] # suffix links for each state

    def train(self, W):
        W = np.array(W)
        if len(W.shape) < 2:
            W = W.reshape(-1,1)
        self.symbols = []
        self.N_features = W.shape[1]
        self.N_states = W.shape[0] + 1
        for col_index in range(W.shape[1]):
            column_data = W[:, col_index]
            unique_vals = np.unique(column_data)
            self.symbols.append(unique_vals.tolist())
        self.S = np.empty((self.N_features, self.N_states), dtype=object)
        self.sigma = []
        for sim in self.symbols:
            self.sigma.append(np.empty((self.N_states, len(sim)), dtype=object))
        for i in range(1,len(W)+1):
            i_word = i-1
            for j in range(self.N_features):
                symbol = W[i_word][j]
                for k in range(len(self.symbols[j])):
                    if self.symbols[j][k] == symbol:
                        symbolIdx = k
                self.sigma[j][i-1,symbolIdx] = i
                k = self.S[j,i-1]
                while k is not None and self.sigma[j][k,symbolIdx] is None:
                    self.sigma[j][k,symbolIdx] = i
                    k = self.S[j,k]
                if k is not None:
                    self.S[j,i] = self.sigma[j][k,symbolIdx]
                else:
                    self.S[j,i] = 0
    def plotOracles(self):
        style = "Simple, tail_width=0.5, head_width=4, head_length=8"
        kw = dict(arrowstyle=style, color="k")
        suffix_style = "Simple, tail_width=0.3, head_width=4, head_length=8"
        kw_suf = dict(arrowstyle=suffix_style, color="k")

        fig, ax = plt.subplots(self.N_features, figsize=(self.N_states+1, self.N_features*3))
        for idx in range(self.N_features):
            # start a new figure
            longest_delta = 1
            for s in range(len(self.S[idx])):
                circle = plt.Circle((0.5+s, 0.5), 0.2, color='black', fill=False)
                ax[idx].add_patch(circle)
                label = ax[idx].annotate(s, xy=(0.5+s, 0.5), fontsize=15, verticalalignment="center", horizontalalignment="center")
                for i in range(len(self.sigma[idx][s])):
                    delta = self.sigma[idx][s][i]
                    if delta == s+1:
                        a = patches.FancyArrowPatch((0.5+s+0.2, 0.5), (0.5+(delta)-0.2, 0.5), **kw)
                        ax[idx].add_patch(a)
                        label = ax[idx].annotate(self.symbols[idx][i], xy=((delta-s)+s, 0.6), fontsize=15, verticalalignment="center", horizontalalignment="center")
                    elif delta is not None:
                        a = patches.FancyArrowPatch((0.5+s, 0.7), (0.5+(delta), 0.7), connectionstyle="arc3,rad=-.5", **kw)
                        ax[idx].add_patch(a)
                        label = ax[idx].annotate(self.symbols[idx][i], xy=((delta-s)/2+s+0.5, 0.9+(delta-s)/4), fontsize=15, verticalalignment="center", horizontalalignment="center")
                        if delta-s > longest_delta:
                            longest_delta = delta-s
                if self.S[idx][s] is not None:
                    a = patches.FancyArrowPatch((0.5+s, 0.3), (0.5+self.S[idx][s], 0.3), connectionstyle="arc3,rad=-.5", **kw_suf, alpha=0.2)
                    ax[idx].add_patch(a)
                    if s-self.S[idx][s] > longest_delta:
                        longest_delta = s-self.S[idx][s]
            ax[idx].set_xlim(0, self.N_states)
            ax[idx].set_ylim(-longest_delta/2+0.5, longest_delta/2+0.5)
            ax[idx].set_aspect('equal', adjustable='box') # Ensures the circle isn't distorted into an ellipse
            ax[idx].axis('off')
        plt.show()

    def predict(self, current_state=0, context_token=None, target_feat=1, p=1):
        i = current_state #if current_state is not None else 0
        src_feat = 0

        # check if there is correspondence with context
        sig = self.sigma[src_feat][i]
        next_available_symbols = [self.symbols[src_feat][j] if sig[j] is not None else None for j in range(len(sig))]
        next_state_from_context = next_available_symbols.index(context_token) if context_token in next_available_symbols else None
        found_match = False
        if next_state_from_context is not None and sig[next_state_from_context] is not None:
            best_state = sig[next_state_from_context]
            sig = self.sigma[target_feat][i]
            next_available_symbols = [self.symbols[target_feat][j] if sig[j] is not None else None for j in range(len(sig))]
            if next_state_from_context is not None and best_state in sig:
                next_state_from_context = sig.tolist().index(best_state)
                v = next_available_symbols[next_state_from_context]
                next_state = best_state
                found_match = True
        elif self.S[target_feat][i] is not None:
            # if best state is not in next states check if it is in a suffix link state
            # print(i)
            # print(self.S[target_feat][1])
            i = self.S[target_feat][i]
            sig = self.sigma[src_feat][i]
            next_available_symbols = [self.symbols[src_feat][j] for j in range(len(sig)) if sig[j] is not None]
            next_state_from_context = next_available_symbols.index(context_token) if context_token in next_available_symbols else None
            if next_state_from_context is not None and sig[next_state_from_context] is not None:
                best_state = sig[next_state_from_context]
                sig = self.sigma[target_feat][i]
                next_available_symbols = [self.symbols[target_feat][j] for j in range(len(sig)) if sig[j] is not None]
                if next_state_from_context is not None and best_state in sig:
                    next_state_from_context = sig.tolist().index(best_state)
                    v = next_available_symbols[next_state_from_context]
                    next_state = best_state
                    found_match = True
                else:
                    found_match = False
        if not found_match:
            i = current_state
            q = 1 if self.S[target_feat][i] == None else p
            if random.random() < q and i < len(self.S[target_feat])-1:
                idx = self.sigma[target_feat][i].tolist().index(i+1)
                v = self.symbols[target_feat][idx]
                i += 1
            else:
                sig = self.sigma[target_feat][self.S[target_feat][i]]
                not_none_idxs = [j for j in range(len(sig)) if sig[j] is not None]
                idx = random.choice(not_none_idxs)
                v = self.symbols[target_feat][idx]
                i = self.sigma[target_feat][self.S[target_feat][i]][idx]
            next_state = i
        return v, next_state

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


    model_load_path = None

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
        print(databases[i])

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
        training_parameters['window_size_s'] = 6
        training_parameters['pred_time_s'] = 0
        training_parameters['seed'] = 666


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


        # LOAD TARGET CORPUS SOUNDFILES FOR CONCATENATIVE SYNTH
        target_corpus_df = pd.read_csv(f'{training_parameters['target_corpus_path']}/corpus_discrete.csv', index_col=0)
        unique_filenames = unique_elements = list(set(target_corpus_df['filename'].values))
        concatenative_corpus = {}
        for audiofilepath in unique_filenames:
            signal, _ = librosa.load(audiofilepath, sr=src_processing_params['sample_rate'], mono=True)
            concatenative_corpus[audiofilepath] = signal


        # SET SEED
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


        src_features = source_corpus_df[features_source].values
        src_scaler = StandardScaler()
        src_scaler.fit(src_features)
        src_normalized_sequences = []
        for seq_df in src_signals_dfs:
            sequence_df = pd.DataFrame(data=src_scaler.transform(seq_df[features_source].values), columns=features_source)
            sequence_df['event_start'] = seq_df['event_start'].values
            src_normalized_sequences.append(sequence_df)

        # normalize target features
        tgt_features = target_corpus_df[features_target].values
        tgt_scaler = StandardScaler()
        tgt_scaler.fit(tgt_features)
        tgt_normalized_sequences = []
        for seq_df in tgt_signals_dfs:
            sequence_df = pd.DataFrame(data=tgt_scaler.transform(seq_df[features_target].values), columns=features_target)
            sequence_df['event_start'] = seq_df['event_start'].values
            tgt_normalized_sequences.append(sequence_df)

        pca = PCA(n_components=2)
        X_scaled = tgt_scaler.transform(tgt_features)
        X_pca = pca.fit_transform(X_scaled)
        brc = Birch(n_clusters=training_parameters['N_clusters_tgt']).fit(X_scaled)

        for seq_df in tgt_normalized_sequences:
            labels = brc.predict(seq_df.values[:,:-1])
            seq_df['labels'] = labels

        tgtAudioSamplesByClass = {}
        for i in range(training_parameters['N_clusters_tgt']):
            tgtAudioSamplesByClass[i] = []


        # make dataloader
        window_size_s = training_parameters['window_size_s']
        pred_time_s = training_parameters['pred_time_s']
        WINDOW_N_SAMPLES = window_size_s * src_processing_params['sample_rate']
        PRED_TIME_SAMPLES = pred_time_s * src_processing_params['sample_rate']
        data = []
        dataByName = {}
        seq_lens = []
        all_src_times = []
        all_tgt_times = []
        for k in range(len(src_normalized_sequences)):
            this_song_data = []
            X_values = src_normalized_sequences[k].values
            y_values = tgt_normalized_sequences[k].values
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
                # x_src = np.concatenate((X_values[indices], x_src_window_times.reshape(-1,1)), axis=1)
                x_src = X_values[indices]
                all_src_times += x_src_window_times.tolist()
                indices = np.where((y_times >= window_start_idx) & (y_times < y_times[j+1]))
                # add time difference index
                x_tgt_window_times = (y_times[j+1] - PRED_TIME_SAMPLES) - y_times[indices]
                # x_tgt = np.concatenate((y_values[indices], x_tgt_window_times.reshape(-1,1)), axis=1)
                x_tgt = y_values[indices]
                all_tgt_times += x_tgt_window_times.tolist()


                tgt_duration = tgt_scaler.inverse_transform(y_tgt[:-2].reshape(1,-1))[:,-1][0]
                y_tgt_audio = concatenative_corpus[f'{training_parameters['sound_corpus_path']}/{common_filenames[k]}/{src_processing_params['target_track_name']}.wav'][int(y_times[j+1]):int(y_times[j+1]+tgt_duration)]
                tgtAudioSamplesByClass[tgt_normalized_sequences[k][['labels']].values[j+1][0]].append(y_tgt_audio)


                data.append([x_src, x_tgt, y_tgt, y_tgt_audio])
                this_song_data.append([x_src, x_tgt, y_tgt, y_tgt_audio])
                seq_lens.append(x_src.shape[0])
                seq_lens.append(x_tgt.shape[0])
            dataByName[common_filenames[k]] = this_song_data


        MAX_SEQ_LEN = int(np.array(seq_lens).mean() + np.array(seq_lens).std())
        if model_load_path:
            MAX_SEQ_LEN = training_parameters['max_sequence_length']
        new_data = []
        for datum in data:
            x_src = datum[0][-MAX_SEQ_LEN:,:]
            x_tgt = datum[1][-MAX_SEQ_LEN:,:]
            new_data.append([x_src, x_tgt, datum[2]])
        data = new_data


        # cluster tgt tokens
        # cluster src sequences of tokens
        c = 0
        data4clustering = []
        for song, songData in dataByName.items():
            src_seq_single = []
            this_data = {}
            for datum in songData:
                src_seq = datum[0]
                tgt_seq = datum[1]
                tgt_objective = datum[2]
                len_src = src_seq.shape[0]
                src_vec_mean = np.mean(src_seq, axis=0).tolist()
                src_vec_std = np.std(src_seq, axis=0).tolist()
                src_feats = src_vec_mean + src_vec_std + [len_src]
                src_seq_single.append(src_feats)
                data4clustering.append(src_feats)
            this_data['data'] = songData
            this_data['src_seq_based_on_target'] = np.array(src_seq_single)
            this_data['tgt_labels'] = tgt_normalized_sequences[c][['labels']].values[1:,:].reshape(-1)
            dataByName[song] = this_data
            c += 1

        data4clustering = np.array(data4clustering)
        brc = Birch(n_clusters=training_parameters['N_clusters_tgt']).fit(data4clustering)
        # print("Explained variance:", pca.explained_variance_ratio_)
        for songName, songData in dataByName.items():
            src_seq4cluster = songData['src_seq_based_on_target']
            labels = brc.predict(src_seq4cluster)
            songData['src_labels'] = labels
            FO = FactorOracle()
            SRC = songData['src_labels'].tolist()
            TGT = songData['tgt_labels'].tolist()
            assert len(SRC) == len(TGT)
            W = [[SRC[i], TGT[i]] for i in range(len(SRC))]
            FO.train(W)
            songData['FO'] = FO
            dataByName[songName] = songData
            # FO.visualize()
        # FO.plotOracles()



        # TEST OVER ALL ELEMENTS IN CORPUS
        tot_accuracy = []
        tot_f1 = []
        tot_mrSTFT = []
        for test_songName in list(dataByName.keys()):
            test_song = dataByName[test_songName]
            predictions = []
            current_state = 0
            FOs = []
            for songName, songData in dataByName.items():
                if songName != test_song:
                    FOs.append(dataByName[songName]['FO'])
            for token in test_song['src_labels'].tolist():
                candidate_results = []
                candidate_next_states = []
                for FO in FOs:
                    tgt_pred, next_state = FO.predict(current_state, token)
                    candidate_results.append(tgt_pred)
                    candidate_next_states.append(next_state)
                from collections import Counter
                counts = Counter(candidate_results)
                vals = list(counts.values())
                if vals[0] > 1:
                    tgt_pred = list(counts.keys())[0]
                    next_state = candidate_next_states[candidate_results.index(tgt_pred)]
                else:
                    idx = random.choice(list(range(len(candidate_results))))
                    tgt_pred = candidate_results[idx]
                    current_state = candidate_next_states[idx]
                predictions.append(tgt_pred)

            # accuracy = test_song['tgt_labels'].tolist() == predictions
            accuracy = [a == b for a, b in zip(test_song['tgt_labels'].tolist(), predictions)]
            accuracy = sum(accuracy) / len(accuracy)

            tot_accuracy.append(accuracy)
            f1 = f1_score(test_song['tgt_labels'].tolist(), predictions, average='weighted')
            tot_f1.append(f1)


            # compute mrSTFT
            mrSTFTs = []
            for ff in range(len(predictions)):
                # datum[3] has the ground truth audio
                pred_sound_segment = random.choice(tgtAudioSamplesByClass[predictions[ff]])
                expected_sound_segment = dataByName[test_songName]['data'][ff][3]
                min_len = min(expected_sound_segment.shape[0], pred_sound_segment.shape[0])
                mrSTFT = spectral_loss(torch.tensor(expected_sound_segment[:min_len]), torch.tensor(pred_sound_segment[:min_len]))
                mrSTFTs.append(mrSTFT.detach().item() if mrSTFT.detach().item() != np.inf and mrSTFT.detach().item() != np.nan else 0)
            mrSTFTs = np.array(mrSTFTs)
            mask = np.isfinite(mrSTFTs)
            mrSTFTs = mrSTFTs[mask]
            tot_mrSTFT.append(np.array(mrSTFTs).mean())

        print(f'tot accuracy: {np.array(tot_accuracy).mean()}')
        print(f'tot f1: {np.array(tot_f1).mean()}')
        print(f'tot mrSTFT: {np.array(tot_mrSTFT).mean()}')
        print()




