import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
import time
from utils.process_data import process_data
from models.dfa import DFA_MLP


def train_DFA(epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    root_path = "/speech/malar/vighnesh/EEG/FeaturesExtracted/SpecDirOneFrame"
    result = process_data(root_path)

    has_separate_test = len(result) == 4

    if has_separate_test:
        X, Y, X_test_raw, Y_test_raw = result
    else:
        X, Y = result

    # Fit encoder on all labels so mapping is consistent
    if has_separate_test:
        all_labels = np.concatenate([Y, Y_test_raw])
    else:
        all_labels = Y

    encoder = LabelEncoder()
    encoder.fit(all_labels)
    Y = encoder.transform(Y)

    # Fit scaler on train only
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if has_separate_test:
        Y_test_raw = encoder.transform(Y_test_raw)
        X_test_raw = scaler.transform(X_test_raw)

        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=Y
        )
        X_test_np = X_test_raw
        Y_test_np = Y_test_raw
    else:
        X_train, X_temp, Y_train, Y_temp = train_test_split(
            X, Y, test_size=0.3, random_state=42, stratify=Y
        )
        X_val, X_test_np, Y_val, Y_test_np = train_test_split(
            X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp
        )

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y_train, dtype=torch.long).to(device)
    Y_val = torch.tensor(Y_val, dtype=torch.long).to(device)
    Y_test = torch.tensor(Y_test_np, dtype=torch.long).to(device)

    input_dim = X_train.shape[1]
    num_classes = len(encoder.classes_)

    print(f"Input dim: {input_dim}, Num classes: {num_classes}")
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    hidden1 = 256
    hidden2 = 256
    lr = 0.01
    batch_size = 32

    model = DFA_MLP(input_dim, hidden1, hidden2, num_classes).to(device)

    # -------- RANDOM DIRECT FEEDBACK MATRICES --------
    B2 = torch.randn(num_classes, hidden2, device=device) / hidden2**0.5
    B1 = torch.randn(num_classes, hidden1, device=device) / hidden1**0.5

    time_start = time.time()

    for epoch in range(epochs):
        perm = torch.randperm(X_train.size(0))
        X_shuf = X_train[perm]
        Y_shuf = Y_train[perm]

        correct = 0
        total = 0

        for i in tqdm(
            range(0, X_train.size(0), batch_size),
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=False
        ):
            xb = X_shuf[i:i + batch_size]
            yb = Y_shuf[i:i + batch_size]

            with torch.no_grad():
                # -------- FORWARD --------
                a1, h1, a2, h2, logits = model(xb)

                logits = logits - logits.max(dim=1, keepdim=True).values
                probs = torch.softmax(logits, dim=1)

                # -------- ONE HOT --------
                y_onehot = torch.zeros_like(probs)
                y_onehot.scatter_(1, yb.unsqueeze(1), 1)

                # -------- OUTPUT ERROR --------
                delta3 = (probs - y_onehot) / xb.size(0)

                # -------- DFA BACKWARD --------
                delta2 = (delta3 @ B2) * (a2 > 0).float()
                delta1 = (delta3 @ B1) * (a1 > 0).float()

                # -------- GRADIENTS --------
                grad3 = delta3.T @ h2
                grad2 = delta2.T @ h1
                grad1 = delta1.T @ xb

                # -------- UPDATE --------
                model.fc3.weight -= lr * grad3
                model.fc2.weight -= lr * grad2
                model.fc1.weight -= lr * grad1

                model.fc3.bias -= lr * delta3.sum(dim=0)
                model.fc2.bias -= lr * delta2.sum(dim=0)
                model.fc1.bias -= lr * delta1.sum(dim=0)

                preds = probs.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        train_acc = correct / total

        # -------- VALIDATION --------
        model.eval()
        with torch.no_grad():
            logits = model(X_val)[-1]
            preds = torch.argmax(logits, dim=1)
            val_acc = (preds == Y_val).float().mean().item()

        print(
            f"Epoch {epoch+1} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}"
        )

    time_end = time.time()

    # -------- TEST --------
    with torch.no_grad():
        logits = model(X_test)[-1]
        preds = torch.argmax(logits, dim=1)
        test_acc = (preds == Y_test).float().mean().item()

    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Time taken: {time_end - time_start:.2f}s")