import os
import json
import shutil
import librosa
import numpy as np
import soundfile as sf

if __name__ == "__main__":

    GENRES = ['singer_songwriter']
    ARTISTS = ['Firefly']
    INSTRUMENTS = ['guitar','piano','bass','drums']
    directory_path = './moisesdb/moisesdb_v0.1'
    destination_dir = 'moisesdb-subsection'
    folders = [entry.name for entry in os.scandir(directory_path) if entry.is_dir()]
    if os.path.exists(destination_dir):
        shutil.rmtree(destination_dir)
    os.makedirs(destination_dir)

    countcopied = 0
    arists = []
    genres = []
    durations = []
    for dir in folders:
        with open(f'{directory_path}/{dir}/data.json', 'r') as f:
            metadata = json.load(f)
        # only get tracks of a specific genre
        if metadata['artist'] in ARTISTS:
            if metadata['genre'] in GENRES:
                arists.append(metadata['artist'])
                genres.append(metadata['genre'])
                subfolders = [entry.name for entry in os.scandir(f'{directory_path}/{dir}') if entry.is_dir()]
                os.makedirs(f'{destination_dir}/{dir}')
                try:
                    if all(element in subfolders for element in INSTRUMENTS):
                        subdirs_to_check = [f'{directory_path}/{dir}/{inst}' for inst in INSTRUMENTS]
                        counter = 0
                        print(dir)
                        for subf in subdirs_to_check:
                            files = [f for f in os.listdir(subf) if f.endswith(".wav")]
                            inst = subf.split('/')[-1]                                    
                            if len(files) == 1:
                                # shutil.copy(f'{subf}/{files[0]}', f'{destination_dir}/{dir}/{inst}.wav')
                                signal, _ = librosa.load(f'{subf}/{files[0]}', sr=44100, mono=True)
                                rms = librosa.feature.rms(y=signal, frame_length=2048, hop_length=1024)
                                if rms.max() < 0.7:
                                    signal *= 0.7/signal.max()
                                sf.write(f'{destination_dir}/{dir}/{inst}.wav', signal, 44100, subtype='PCM_24')
                                duration_seconds = signal.shape[0] / 44100
                                durations.append(duration_seconds)
                            elif inst in ['guitar', 'bass', 'piano']:
                                signal, _ = librosa.load(f'{subf}/{files[0]}', sr=44100, mono=True)
                                rms = librosa.feature.rms(y=signal, frame_length=2048, hop_length=1024)
                                if rms.max() < 0.7:
                                    signal *= 0.7/signal.max()
                                sf.write(f'{destination_dir}/{dir}/{inst}.wav', signal, 44100, subtype='PCM_24')
                                duration_seconds = signal.shape[0] / 44100
                                durations.append(duration_seconds)
                            else:
                                # sum signals in case of drums
                                signals = []
                                signal, _ = librosa.load(f'{subf}/{files[0]}', sr=44100, mono=True)
                                sig_shape = signal.shape[0] 
                                for f in files:
                                    signal, _ = librosa.load(f'{subf}/{f}', sr=44100, mono=True)
                                    if signal.shape[0] == sig_shape:
                                        signals.append(signal)
                                signals = np.array(signals)
                                signals = np.sum(signals, axis=0) / (signals.shape[0]/2)
                                rms = librosa.feature.rms(y=signals, frame_length=2048, hop_length=1024)
                                if rms.max() < 0.7:
                                    signals *= 0.7/signals.max()
                                sf.write(f'{destination_dir}/{dir}/{inst}.wav', signals, 44100, subtype='PCM_24')
                                duration_seconds = signals.shape[0] / 44100
                                durations.append(duration_seconds)
                        with open(f'{destination_dir}/{dir}/data.json', 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, ensure_ascii=False, indent=4)
                        countcopied += 1
                except:
                    print(f"Couldn't copy song {dir}")
                    shutil.rmtree(f'{destination_dir}/{dir}')
    print(f'copied {countcopied} songs')

    from collections import Counter

    item_counts = Counter(arists)
    print(f'artists: {item_counts}')
    item_counts = Counter(genres)
    print(f'genres: {item_counts}')
    print(f'total_duration: {np.array(durations).sum()} seconds')
    print(f'total_duration: {np.array(durations).sum()/60} minutes')
    print(f'total_duration: {np.array(durations).sum()/3600} hours')