import pandas as pd
import librosa
import matplotlib.pyplot as plt
import json


def plotFromDB(corpus_path, 
				soundfile_path,
				plot_feature='rms_mean', 
				sample_start_s=0, 
				sample_duration_s=None,
				save_dir=None,
				showfig=False):

	with open(f'{corpus_path}/featureExtraction_params.json', 'r') as f:
		processing_params = json.load(f)
	sample_rate = processing_params['sample_rate']

	corpus_df_path = f'{corpus_path}/corpus_discrete.csv'
	source_corpus_df = pd.read_csv(corpus_df_path, index_col=0)

	plot_df_src = source_corpus_df[source_corpus_df['filename'] == soundfile_path]
	signal, _ = librosa.load(soundfile_path, sr=sample_rate, mono=True)

	sample_start = sample_start_s * sample_rate
	if sample_duration_s:
		sample_duration = sample_duration_s * sample_rate

	plot_feature_src = plot_feature
	plot_df_src[plot_feature_src] = 2*((plot_df_src[plot_feature_src] - plot_df_src[plot_feature_src].min())/(plot_df_src[plot_feature_src].max() - plot_df_src[plot_feature_src].min())) - 1
	plot_df_src = plot_df_src[plot_df_src['event_start'] > sample_start]
	if sample_duration_s and (sample_start+sample_duration) < signal.shape[0]:
		plot_df_src = plot_df_src[plot_df_src['event_start'] < sample_start+sample_duration]
	plot_df_src['event_start'] = plot_df_src['event_start'] - sample_start

	fig, ax = plt.subplots(1, figsize=(20, 4))
	if sample_duration_s and (sample_start+sample_duration) < signal.shape[0]:
		librosa.display.waveshow(y=signal[sample_start:sample_start+sample_duration], sr=sample_rate, color="black", x_axis='time', ax=ax, alpha=0.2)
	elif sample_start < signal.shape[0]:
		librosa.display.waveshow(y=signal[sample_start:], sr=sample_rate, color="black", x_axis='time', ax=ax, alpha=0.2)
	ax.scatter(librosa.samples_to_time(plot_df_src['event_start'], sr=sample_rate), 
				y=plot_df_src[plot_feature_src], label=plot_feature_src, edgecolors='r', facecolors='none')
	ax.hlines(y=plot_df_src[plot_feature_src],
			xmin=librosa.samples_to_time(plot_df_src['event_start'], sr=sample_rate), 
			xmax=librosa.samples_to_time(plot_df_src['event_start']+plot_df_src['event_duration'], sr=sample_rate), 
			linewidth=1, label='event duration', colors='r')
	ax.vlines(librosa.samples_to_time(plot_df_src['event_start'], sr=sample_rate), 
		-1, 1, color='r', alpha=0.1, linestyle='--', label='onsets', linewidth=0.5)
	ax.legend()
	ax.set_xlabel('time [s]')
	ax.set_ylabel(plot_feature_src)
	ax.set_title(soundfile_path)
	if save_dir:
		plt.savefig(f'{save_dir}.png')
	if showfig:
		plt.show()
	return


def plotCoupleFromDB(src_corpus_path,
					tgt_corpus_path, 
					src_audiofile_path, 
					tgt_audiofile_path,
					sample_rate=44100,
					plot_feature_src='rms_mean', 
					plot_feature_tgt='rms_mean', 
					sample_start_s=0, 
					sample_duration_s=None,
					save_dir=None,
					showfig=False):

	# with open(f'{src_corpus_path}/featureExtraction_params.json', 'r') as f:
	# 	processing_params = json.load(f)
	# sample_rate = processing_params['sample_rate']
	# sample_rate = 44100

	# corpus_df_path_src = f'{src_corpus_path}/corpus_discrete.csv'
	corpus_df_path_src = src_corpus_path
	source_corpus_df = pd.read_csv(corpus_df_path_src, index_col=0)
	# corpus_df_path_tgt = f'{tgt_corpus_path}/corpus_discrete.csv'
	corpus_df_path_tgt = tgt_corpus_path
	target_corpus_df = pd.read_csv(corpus_df_path_tgt, index_col=0)

	plot_df_src = source_corpus_df[source_corpus_df['filename'] == src_audiofile_path]
	signal_src, _ = librosa.load(src_audiofile_path, sr=sample_rate, mono=True)

	plot_df_tgt = target_corpus_df[target_corpus_df['filename'] == tgt_audiofile_path]
	signal_tgt, _ = librosa.load(tgt_audiofile_path, sr=sample_rate, mono=True)

	sample_start = sample_start_s * sample_rate
	if sample_duration_s:
		sample_duration = sample_duration_s * sample_rate

	plot_df_src[plot_feature_src] = 2*((plot_df_src[plot_feature_src] - plot_df_src[plot_feature_src].min())/(plot_df_src[plot_feature_src].max() - plot_df_src[plot_feature_src].min())) - 1
	plot_df_src = plot_df_src[plot_df_src['event_start'] > sample_start]
	if sample_duration_s:
		plot_df_src = plot_df_src[plot_df_src['event_start'] < sample_start+sample_duration]
	plot_df_src['event_start'] = plot_df_src['event_start'] - sample_start

	plot_df_tgt[plot_feature_tgt] = 2*((plot_df_tgt[plot_feature_tgt] - plot_df_tgt[plot_feature_tgt].min())/(plot_df_tgt[plot_feature_tgt].max() - plot_df_tgt[plot_feature_tgt].min())) - 1
	plot_df_tgt = plot_df_tgt[plot_df_tgt['event_start'] > sample_start]
	if sample_duration_s:
		plot_df_tgt = plot_df_tgt[plot_df_tgt['event_start'] < sample_start+sample_duration]
	plot_df_tgt['event_start'] = plot_df_tgt['event_start'] - sample_start

	fig, ax = plt.subplots(2, figsize=(20, 8))
	if sample_duration_s and (sample_start+sample_duration) < signal_src.shape[0] and (sample_start+sample_duration) < signal_tgt.shape[0]:
		librosa.display.waveshow(y=signal_src[sample_start:sample_start+sample_duration], sr=sample_rate, color="black", x_axis='time', ax=ax[0], alpha=0.2)
		librosa.display.waveshow(y=signal_tgt[sample_start:sample_start+sample_duration], sr=sample_rate, color="black", x_axis='time', ax=ax[1], alpha=0.2)
	elif sample_start < signal_src.shape[0] and sample_start < signal_tgt.shape[0]:
		librosa.display.waveshow(y=signal_src[sample_start:], sr=sample_rate, color="black", x_axis='time', ax=ax[0], alpha=0.2)
		librosa.display.waveshow(y=signal_tgt[sample_start:], sr=sample_rate, color="black", x_axis='time', ax=ax[1], alpha=0.2)
	ax[0].scatter(librosa.samples_to_time(plot_df_src['event_start'], sr=sample_rate), 
				y=plot_df_src[plot_feature_src], label=plot_feature_src, edgecolors='r', facecolors='none')
	ax[1].scatter(librosa.samples_to_time(plot_df_tgt['event_start'], sr=sample_rate), 
				y=plot_df_tgt[plot_feature_tgt], label=plot_feature_tgt, edgecolors='b', facecolors='none')
	ax[0].hlines(y=plot_df_src[plot_feature_src],
			xmin=librosa.samples_to_time(plot_df_src['event_start'], sr=sample_rate), 
			xmax=librosa.samples_to_time(plot_df_src['event_start']+plot_df_src['event_duration'], sr=sample_rate), 
			linewidth=1, label='event duration', colors='r')
	ax[1].hlines(y=plot_df_tgt[plot_feature_tgt],
			xmin=librosa.samples_to_time(plot_df_tgt['event_start'], sr=sample_rate), 
			xmax=librosa.samples_to_time(plot_df_tgt['event_start']+plot_df_tgt['event_duration'], sr=sample_rate), 
			linewidth=1, label='event duration', colors='b')
	ax[0].vlines(librosa.samples_to_time(plot_df_src['event_start'], sr=sample_rate), 
		-1, 1, color='r', alpha=0.1, linestyle='--', label='onsets', linewidth=0.5)
	ax[1].vlines(librosa.samples_to_time(plot_df_tgt['event_start'], sr=sample_rate), 
		-1, 1, color='r', alpha=0.1, linestyle='--', label='onsets', linewidth=0.5)
	ax[0].legend()
	ax[1].legend()
	ax[0].set_xlabel('time [s]')
	ax[1].set_xlabel('time [s]')
	ax[0].set_ylabel(plot_feature_src)
	ax[1].set_ylabel(plot_feature_tgt)
	ax[0].set_ylabel(plot_feature_src)
	ax[1].set_ylabel(plot_feature_tgt)
	ax[0].set_title(src_audiofile_path.split('/')[-1].split('.')[0])
	ax[1].set_title(tgt_audiofile_path.split('/')[-1].split('.')[0])
	ax[0].set_ylim(-1.2, 1.2)
	ax[1].set_ylim(-1.2, 1.2)
	# plt.suptitle('Make my day - Firefly')
	plt.suptitle(src_audiofile_path.split('/')[-2])
	fig.tight_layout()
	if save_dir:
		plt.savefig(f'{save_dir}.png')
	if showfig:
		plt.show()

	return



if __name__ == "__main__":


	corpus_path = '00_corpus/moisesdb/moisesdb-BassGuit4figs'
	song = '6c70d5e0-5972-444a-86f8-a558dbb92d92'
	src_track_name = 'bass'
	tgt_track_name = 'guitar'
	plotCoupleFromDB(f'{corpus_path}/00_process_src/corpus_discrete.csv', 
					f'{corpus_path}/00_process_tgt/corpus_discrete.csv', 
					f'{corpus_path}/{song}/{src_track_name}.wav',
					f'{corpus_path}/{song}/{tgt_track_name}.wav',
					sample_rate=44100,
					plot_feature_src='rms_mean', 
					plot_feature_tgt='rms_mean', 
					sample_start_s=20, 
					sample_duration_s=30,
					save_dir=f'{corpus_path}/{song}/{src_track_name}_{tgt_track_name}')

