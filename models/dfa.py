import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
import time
from utils.process_data import process_data



class DFA_MLP(nn.Module):

    def __init__(self, input_dim, hidden1, hidden2, num_classes):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):

        a1 = self.fc1(x)
        h1 = torch.relu(a1)

        a2 = self.fc2(h1)
        h2 = torch.relu(a2)

        logits = self.fc3(h2)

        return a1, h1, a2, h2, logits