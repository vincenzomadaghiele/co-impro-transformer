# Modeling relations between musical events in continuous time with transformer models for live co-improvisational interactions

Supplementary materials for this paper, including audio and video examples, can be found at [the companion page](https://vincenzomadaghiele.github.io/co-impro-transformer-AIMC/)


## 1. Install dependencies
Download and install uv from [here](https://docs.astral.sh/uv/guides/install-python/)

## 2. Train and play with a custom model
Insert paired tracks of corpus in the folder as `./00_corpus/<corpus-name>/<track-name>/source.wav` and `./00_corpus/<corpus-name>/<track-name>/target.wav`.

Process data: 
```
uv run dataProcess.py
```

Train the model: 
```
uv run transformerT2V_train.py
```

Play with the model: 
```
uv run transformerT2V_live.py
```

Fine-tune model with recordings: 
```
uv run transformerT2V_finetune.py
```

Arguments for each of these python scripts have to be insterted in the scripts, in the dictionary at the beginning of the main method.