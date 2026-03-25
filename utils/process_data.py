from pathlib import Path
from tqdm import tqdm
import numpy as np

def process_data(root_path):
    folder = Path(root_path)

    X = []
    Y = []

    for file in tqdm(folder.iterdir(), desc="Loading files"):

        filename = file.stem

        if filename == "newSourceFileList":
            continue

        audio_person, utterance_index, syllable_channel = filename.split("_")
        syllable, channel = syllable_channel.split("col")

        with open(file) as f:
            for line in f:

                arr = np.fromstring(line, sep=" ")

                if arr.size == 0:
                    continue

                X.append(arr)
                Y.append(syllable)

    X = np.array(X)
    Y = np.array(Y)
    
    print(np.unique(Y))
    return X,Y