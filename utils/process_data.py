from pathlib import Path
from tqdm import tqdm
import numpy as np


def process_data(root_path):
    print("Select mode:")
    print("  1 - All channels")
    print("  2 - Temporal + Parietal only")
    print("  3 - Leave-one-subject-out (Subject 1 as test)")
    mode = int(input("Enter mode (1/2/3): "))

    folder = Path(root_path)
    X = []
    Y = []
    X_test = []
    Y_test = []

    temporal = [44, 39, 40, 45, 46, 50, 56, 57, 109, 115, 114, 108, 102, 101, 100, 107]
    parietal = [47, 51, 52, 53, 54, 58, 59, 60, 61, 67, 77, 78, 79, 85, 86, 91, 92, 96, 97, 98]
    relevant_channels = set(temporal + parietal)

    for file in tqdm(folder.iterdir(), desc="Loading files"):
        filename = file.stem
        if filename == "newSourceFileList":
            continue

        audio_person, utterance_index, syllable_channel = filename.split("_")
        syllable, channel = syllable_channel.split("col")
        channel_num = int(channel)

        audio_subject = audio_person[1]
        if audio_subject == 'F':
            audio_subject = '5'

        match mode:
            case 1:
                pass
            case 2:
                if channel_num not in relevant_channels:
                    continue
            case 3:
                pass
            case _:
                raise ValueError(f"Invalid mode: {mode}. Use 1, 2, or 3.")

        with open(file) as f:
            for line in f:
                arr = np.fromstring(line, sep=" ")
                if arr.size == 0:
                    continue

                if mode == 3 and audio_subject == '5':
                    X_test.append(arr)
                    Y_test.append(syllable)
                else:
                    X.append(arr)
                    Y.append(syllable)

    X = np.array(X)
    Y = np.array(Y)

    mode_names = {1: "All channels", 2: "Temporal + Parietal", 3: "Leave-one-subject-out"}
    print(f"Mode: {mode_names.get(mode, 'Unknown')}")
    print(f"Classes: {np.unique(Y)}")
    print(f"Train samples: {len(X)}, Labels: {len(Y)}")

    if mode == 3:
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        print(f"Test samples: {len(X_test)}, Labels: {len(Y_test)}")
        print(f"Test classes: {np.unique(Y_test)}")
        return X, Y, X_test, Y_test

    return X, Y