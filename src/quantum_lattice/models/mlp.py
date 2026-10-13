"""The paper's local three-observable magnitude regressor.

This is a PyTorch adaptation of ``Reference Paper Codes/ANN/network.py``.
The source uses TensorFlow's Dense defaults: Glorot-uniform weights and zero
biases. Inputs and linear predictions are deliberately left untransformed.
"""

from __future__ import annotations

import torch
from torch import nn


class PaperMLP(nn.Module):
    """Map ``(<A_u>, <A_v>, <A_u A_v>)`` to a field magnitude prediction."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 150),
            nn.ReLU(),
            nn.Linear(150, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != 3:
            raise ValueError("MLP inputs must have shape (rows, 3)")
        return self.layers(inputs)
