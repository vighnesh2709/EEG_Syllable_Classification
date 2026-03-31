from pathlib import Path
from tqdm import tqdm
import numpy as np

def process_data(root_path):
    folder = Path(root_path)

    X = []
    Y = []
    X_test = []
    Y_test = []

    temporal = [44, 39, 40, 45, 46, 50, 56, 57, 109, 115, 114, 108, 102, 101, 100, 107]
    parietal = [47, 51, 52, 53, 54, 58, 59, 60, 61, 67, 77, 78, 79, 85, 86, 91, 92, 96, 97, 98]

    # temporal = [1]
    # parietal = [3]

    check_temporal = {}
    check_parietal = {}
    no_brain = {}

    count = 0

    for file in tqdm(folder.iterdir(), desc="Loading files"):

        filename = file.stem

        if filename == "newSourceFileList":
            continue

        audio_person, utterance_index, syllable_channel = filename.split("_")
        syllable, channel = syllable_channel.split("col")
        

        audio_subject = audio_person[1]
        if audio_subject == 'F':
            audio_subject = '5'

        with open(file) as f:
            for line in f:
                arr = np.fromstring(line, sep=" ")

                if arr.size == 0:
                    continue

                X.append(arr)
                Y.append(syllable)
                
        

    X = np.array(X)
    Y = np.array(Y)
    # X_test = np.array(X_test)
    # Y_test = np.array(Y_test)
    
    print(np.unique(Y))
    print(len(X),len(Y))
    print(len(X_test),len(Y_test))
    print(f"Temporal: {len(temporal)}   check: {len(check_temporal)}")
    print(f"Parietal: {len(parietal)}   check: {len(check_parietal)}")
    print(f"No Brain: {128 - (len(temporal) + len(parietal))}   check: {len(no_brain)}")
    return X,Y