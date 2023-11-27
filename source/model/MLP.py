import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_features: int = None,
                 out_features: int = None,
                 drop: float = 0.
                 ):
        """
        Initializes the MLP module.

        Args:
        - in_features (int): Number of input features.
        - hidden_features (int, optional): Number of hidden features. Defaults to None (uses in_features).
        - out_features (int, optional): Number of output features. Defaults to None (uses in_features).
        - drop (float, optional): Dropout probability. Defaults to 0.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP module.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after passing through the MLP.
        """
        x = self.nn(x)

        return x