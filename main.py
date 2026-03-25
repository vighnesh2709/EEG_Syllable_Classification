import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
import time
from utils.process_data import process_data

class MLP(nn.Module):

    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    root_path = "/speech/malar/vighnesh/EEG/FeaturesExtracted/SpecDirOneFrame"

    X,Y = process_data(root_path)

    encoder = LabelEncoder()
    Y = encoder.fit_transform(Y)
    print(np.unique(Y))

    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=0.3, random_state=41
    )

    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, random_state=41
    )

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    Y_train = torch.tensor(Y_train, dtype=torch.long).to(device)
    Y_val = torch.tensor(Y_val, dtype=torch.long).to(device)
    Y_test = torch.tensor(Y_test, dtype=torch.long).to(device)

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(Y))

    model = MLP(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    batch_size = 64


    time_start = time.time()

    for epoch in tqdm(range(epochs), desc="Training"):

        model.train()

        perm = torch.randperm(X_train.size(0))

        correct = 0
        total = 0

        for i in range(0, X_train.size(0), batch_size):

            idx = perm[i:i+batch_size]

            batch_x = X_train[idx]
            batch_y = Y_train[idx]

            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)

            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        train_acc = correct / total


        model.eval()

        with torch.no_grad():

            outputs = model(X_val)

            preds = torch.argmax(outputs, dim=1)

            val_correct = (preds == Y_val).sum().item()
            val_total = Y_val.size(0)

            val_acc = val_correct / val_total

        tqdm.write(
            f"Epoch {epoch+1}: Train Acc = {train_acc:.4f} | Val Acc = {val_acc:.4f}"
        )

    time_end = time.time()
    model.eval()

    with torch.no_grad():

        outputs = model(X_test)

        preds = torch.argmax(outputs, dim=1)

        test_correct = (preds == Y_test).sum().item()
        test_total = Y_test.size(0)

        test_acc = test_correct / test_total

    print("Final Test Accuracy:", test_acc)
    print("Time taken: ", time_end - time_start)


if __name__ == "__main__":
    main()
