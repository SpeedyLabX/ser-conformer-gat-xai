"""Classifier head placeholder: BiGRU / pooling / FC"""
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, n_classes=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        return self.fc(x)
