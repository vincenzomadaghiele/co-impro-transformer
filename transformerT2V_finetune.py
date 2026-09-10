import json
import os
from transformerT2V_train import trainTransformerDiscreteT2V
from transformerT2V_evaluate import renderModelOutput

if __name__ == "__main__":

	save_dir = '01_model_logs'
	model_dir = '01_model_logs/2026-01-07/1767777610-transformerDiscreteT2V'
	with open(f'{model_dir}/training_config.json', 'r') as f:
		training_parameters = json.load(f)
	model_load_path = f'{model_dir}/best_model.pth'

	# change training info (new dataset, lr...)
	training_parameters['source_corpus_path'] = '00_corpus/musdb-subset/00_process_src'
	training_parameters['target_corpus_path'] = '00_corpus/musdb-subset/00_process_tgt'
	training_parameters['sound_corpus_path'] = '00_corpus/musdb-subset'
	training_parameters['dataset_training_percentage'] = 0.7
	training_parameters['batch_size'] = 16
	training_parameters['learning_rate'] = 0.001
	training_parameters['epochs'] = 700

	# TRAIN MODEL
	model_logdir = trainTransformerDiscreteT2V(training_parameters, 
											save_dir, 
											model_load_path=model_load_path)

	# EVALUATE MODEL
	N_tracks = 4
	corpus_files = os.listdir(training_parameters['sound_corpus_path'])
	corpus_files = [filename for filename in corpus_files if os.path.isdir(f'{training_parameters['sound_corpus_path']}/{filename}') and filename != '00_process_src' and filename != '00_process_tgt']
	eval_track_names = corpus_files[:4]
	renderModelOutput(model_logdir, eval_track_names)
