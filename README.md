# Modeling relations between musical events in continuous time with transformer models for live co-improvisational interactions

Supplementary materials for this paper, including audio and video examples, can be found at [the companion page](https://vincenzomadaghiele.github.io/co-impro-transformer-AIMC/)

This paper presents a live semi-autonomous system that co-improvises with a live musician using a model learned from the relationship observed in paired recordings of an improvising duo. This responsive system is based on a deep learning approach that models the relations between sequences of events produced by co-playing musicians in continuous time, using transformer models. We present the architecture for simultaneous sequences of sonic events and provide a quantitative evaluation on several representative tasks. Results are compared against a canonical transformer and a multichannel Factor Oracle, the latter being a widely used model for live sequence‑based symbolic music generation. We detail a live implementation employing concatenative synthesis and introduce a customization procedure in which the generative model is iteratively retrained on curated, satisfactory sections from sound recordings of actual human-system co-improvisational interactions. This iterative fine-tuning enables the model's stylistic output to diverge from the original training corpus. Two musical use cases demonstrate the application of this technique. 

More information in the paper ([bib](./seq2seqAIMC.bib)):
> Vincenzo Madaghiele, Pasquale Lisena, Raphaël Troncy.
> [**Modeling Relations Between Musical Events in Continuous Time with Transformer Models for Live Co-Improvisational Interactions**](https://zenodo.org/records/22276042).
> In _7th Conference on AI Music Creativity (AIMC 2026)_, 16-18 September 2026, Staatliches Institut für Musikforschung, Berlin.


## 1. Install dependencies
```
venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Train and play with a custom model
Insert paired tracks of corpus in the folder as `./00_corpus/<corpus-name>/<track-name>/source.wav` and `./00_corpus/<corpus-name>/<track-name>/target.wav`. Arguments for each of these python scripts have to be insterted in the scripts, in the dictionary at the beginning of the main method.

Process data: 
```
python3 dataProcess.py
```

Train the model: 
```
python3 transformerT2V_train.py
```

Play with the model: 
```
python3 transformerT2V_live.py
```

Fine-tune model with recordings: 
```
python3 transformerT2V_finetune.py
```