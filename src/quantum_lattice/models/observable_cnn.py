"""Periodic observable-map models for a controlled square-lattice comparison.

Inputs have channels ``(A, AA_horizontal, AA_vertical)`` and shape
``(batch, 3, L, L)``. At cell ``(y, x)``, the pair channels refer to the
stars at ``(y, x)`` and ``(y, x + 1)`` or ``(y + 1, x)``, respectively,
with periodic indices. Outputs are horizontal/vertical edge-field magnitudes
in shape ``(batch, 2, L, L)``. These are raw linear regression predictions;
neither inputs nor predictions are clipped.

This module defines architectures only. It does not select data, train models,
move tensors to a device, or read saved experiment outputs.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from quantum_lattice.models.mlp import PaperMLP


def _validate_maps(inputs: torch.Tensor) -> None:
    if not isinstance(inputs, torch.Tensor) or not torch.is_floating_point(inputs):
        raise TypeError("Observable maps must be real floating-point torch tensors")
    if inputs.ndim != 4 or inputs.shape[1] != 3:
        raise ValueError("Observable maps must have shape (batch, 3, L, L)")
    if inputs.shape[0] < 1 or inputs.shape[2] < 3 or inputs.shape[2] != inputs.shape[3]:
        raise ValueError("Observable maps require a nonempty batch and square lattice L >= 3")


class PeriodicObservableCNN(nn.Module):
    """A fully convolutional, translation-equivariant observable regressor.

    Defaults give three circular-padded 3x3 hidden convolutions and a linear
    1x1 output layer, with 37,922 trainable parameters. The receptive field
    is 7x7 before periodic wraparound: on L=3 or L=4 it wraps around the
    lattice repeatedly, rather than supplying 49 distinct positions.
    A ``kernel_size=1`` variant is a
    coordinate-local channel-mixing diagnostic, not the paper's three-feature
    edge MLP. ``hidden_channels`` controls both hidden widths and layer count.
    """

    def __init__(
        self,
        hidden_channels: Sequence[int] = (32, 64, 32),
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        widths = tuple(hidden_channels)
        if not widths or any(isinstance(width, bool) or not isinstance(width, int) or width < 1
                             for width in widths):
            raise ValueError("hidden_channels must contain positive integer widths")
        if isinstance(kernel_size, bool) or not isinstance(kernel_size, int) or kernel_size not in (1, 3):
            raise ValueError("kernel_size must be 1 or 3")
        self.hidden_channels = widths
        self.kernel_size = kernel_size
        layers: list[nn.Module] = []
        previous = 3
        for width in widths:
            layers.extend((
                nn.Conv2d(previous, width, kernel_size, padding=kernel_size // 2,
                          padding_mode="circular"),
                nn.ReLU(),
            ))
            previous = width
        layers.append(nn.Conv2d(previous, 2, 1))
        self.layers = nn.Sequential(*layers)
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _validate_maps(inputs)
        return self.layers(inputs)


class ObservableMapMLP(nn.Module):
    """Apply one unchanged :class:`PaperMLP` at every horizontal/vertical edge.

    Every local input is exactly ``(A_u, A_v, AA_uv)``. Rows follow canonical
    edge order ``2 * (y * L + x) + orientation`` (horizontal=0, vertical=1).
    No new learnable layers are added. Passing ``mlp`` reuses its weights and
    lets existing local-regressor predictions be checked against map output.
    """

    def __init__(self, mlp: PaperMLP | None = None) -> None:
        super().__init__()
        if mlp is not None and not isinstance(mlp, PaperMLP):
            raise TypeError("mlp must be a PaperMLP instance or None")
        self.mlp = PaperMLP() if mlp is None else mlp

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _validate_maps(inputs)
        batch, _, size, _ = inputs.shape
        stars = inputs[:, 0]
        horizontal = torch.stack((stars, torch.roll(stars, shifts=-1, dims=-1),
                                  inputs[:, 1]), dim=-1)
        vertical = torch.stack((stars, torch.roll(stars, shifts=-1, dims=-2),
                                inputs[:, 2]), dim=-1)
        rows = torch.stack((horizontal, vertical), dim=-2).reshape(-1, 3)
        return self.mlp(rows).reshape(batch, size, size, 2).permute(0, 3, 1, 2)
