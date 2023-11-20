import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        # Linear Layers
        self.nn = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=int(hidden_features)),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(in_features=int(hidden_features), out_features=out_features),
            nn.LeakyReLU(),
            nn.Dropout(p=drop),
        )

        # Activation(s)

    def forward(self, x):
        x = self.nn(x)

        return x