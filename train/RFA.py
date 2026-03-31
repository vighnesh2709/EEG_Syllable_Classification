import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
import time
from utils.process_data import process_data
from models.rfa import RFA_MLP




def train_RFA():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    root_path = "/speech/malar/vighnesh/EEG/FeaturesExtracted/SpecDirOneFrame"
    folder = Path(root_path)

    X,Y = process_data(root_path)

    encoder = LabelEncoder()
    Y = encoder.fit_transform(Y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split dataset
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )

    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, random_state=42
    )

    # tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    Y_train = torch.tensor(Y_train, dtype=torch.long).to(device)
    Y_val = torch.tensor(Y_val, dtype=torch.long).to(device)
    Y_test = torch.tensor(Y_test, dtype=torch.long).to(device)

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(Y))

    hidden1 = 256
    hidden2 = 256
    lr = 0.01
    batch_size = 32
    epochs = 50

    model = RFA_MLP(input_dim, hidden1, hidden2, num_classes).to(device)

    # -------- RANDOM FEEDBACK MATRICES --------
    B3 = torch.randn(num_classes, hidden2, device=device) / hidden2**0.5
    B2 = torch.randn(hidden2, hidden1, device=device) / hidden1**0.5

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

            xb = X_shuf[i:i+batch_size]
            yb = Y_shuf[i:i+batch_size]

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

                # -------- RFA BACKWARD --------
                delta2 = (delta3 @ B3) * (a2 > 0).float()
                delta1 = (delta2 @ B2) * (a1 > 0).float()

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

    # -------- TEST --------
    time_end = time.time()
    with torch.no_grad():

        logits = model(X_test)[-1]
        preds = torch.argmax(logits, dim=1)

        test_acc = (preds == Y_test).float().mean().item()

    print("Final Test Accuracy:", test_acc)
    print("Time taken: ", time_end - time_start)



# if __name__ == "__main__":
#     main()