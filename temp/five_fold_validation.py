import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
from utils.process_data import process_data
from sklearn.model_selection import KFold

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
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    root_path = "/speech/malar/vighnesh/EEG/FeaturesExtracted/SpecDirOneFrame"
    
    X, Y = process_data(root_path)

    # --- 2. Preprocessing ---
    encoder = LabelEncoder()
    Y_encoded = encoder.fit_transform(Y)
    num_classes = len(encoder.classes_)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 3. 5-Fold CV Setup ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    # Iterate through the 5 folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        print(f"\n--- Starting Fold {fold + 1}/5 ---")
        
        # Split data for this fold
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        Y_train, Y_val = Y_encoded[train_idx], Y_encoded[val_idx]

        # Convert to DataLoaders
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                 torch.tensor(Y_train, dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                               torch.tensor(Y_val, dtype=torch.long))
        
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64)

        # IMPORTANT: Re-initialize model and optimizer for EVERY fold
        model = MLP(X_train.shape[1], num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # --- 4. Training Loop for this fold ---
        epochs = 50 # Reduced for CV efficiency
        best_fold_acc = 0

        for epoch in range(epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation for this fold
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    preds = torch.argmax(outputs, dim=1)
                    val_correct += (preds == batch_y).sum().item()
                    val_total += batch_y.size(0)
            
            fold_acc = val_correct / val_total
            if fold_acc > best_fold_acc:
                best_fold_acc = fold_acc

        print(f"Fold {fold+1} Best Accuracy: {best_fold_acc:.4f}")
        fold_results.append(best_fold_acc)

    # --- 5. Final Statistics ---
    print("\n" + "="*30)
    print(f"5-Fold Mean Accuracy: {np.mean(fold_results):.4f}")
    print(f"Standard Deviation: {np.std(fold_results):.4f}")
    print("="*30)

if __name__ == "__main__":
    main()