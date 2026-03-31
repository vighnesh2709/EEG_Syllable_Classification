import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
import time
from utils.process_data import process_data
from models.mlp import MLP




def train_MLP():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    root_path = "/speech/malar/vighnesh/EEG/FeaturesExtracted/SpecDirOneFrame"
    X, Y = process_data(root_path)

    encoder = LabelEncoder()
    Y = encoder.fit_transform(Y)
    print(np.unique(Y))

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=0.3, random_state=41, stratify=Y  # stratify keeps class balance
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, random_state=41, stratify=Y_temp
    )

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y_train, dtype=torch.long).to(device)
    Y_val = torch.tensor(Y_val, dtype=torch.long).to(device)
    Y_test = torch.tensor(Y_test, dtype=torch.long).to(device)

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(Y))

    model = MLP(input_dim, num_classes, dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Cosine annealing LR scheduler
    epochs = 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    batch_size = 64

    # Early stopping setup
    best_val_acc = 0.0
    patience = 20
    patience_counter = 0
    best_model_state = None

    time_start = time.time()

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        perm = torch.randperm(X_train.size(0))
        correct = 0
        total = 0

        for i in range(0, X_train.size(0), batch_size):
            idx = perm[i:i + batch_size]
            batch_x = X_train[idx]
            batch_y = Y_train[idx]

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        scheduler.step()
        train_acc = correct / total

        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            preds = torch.argmax(outputs, dim=1)
            val_acc = (preds == Y_val).sum().item() / Y_val.size(0)

        tqdm.write(
            f"Epoch {epoch+1}: Train Acc = {train_acc:.4f} | Val Acc = {val_acc:.4f} | LR = {scheduler.get_last_lr()[0]:.6f}"
        )

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    time_end = time.time()

    # Load best model for final test
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        preds = torch.argmax(outputs, dim=1)
        test_acc = (preds == Y_test).sum().item() / Y_test.size(0)

    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Time taken: {time_end - time_start:.2f}s")


# if __name__ == "__main__":
#     main()