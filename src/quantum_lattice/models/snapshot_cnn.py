"""Matched periodic CNNs for raw Z snapshots and their complete-edge means.

Both models contain exactly the same encoder/head and 38,690 parameters. The
mean baseline applies the nonlinear encoder after averaging shots. The set model
applies it to each shot before an arithmetic mean over the shot axis. No labels,
known fields, shot ordering, normalization, or fitted estimator enter either
forward pass. These models predict magnitudes; they do not recover field signs.
"""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


METHODS = ("mean_cnn", "set_cnn")


def _options(method: str, shot_chunk_size: int, activation_checkpoint: bool) -> None:
    if method not in METHODS:
        raise ValueError(f"Unknown snapshot method {method!r}; choose from {METHODS}")
    if type(shot_chunk_size) is not int or shot_chunk_size < 1:
        raise ValueError("shot_chunk_size must be a positive integer")
    if type(activation_checkpoint) is not bool:
        raise ValueError("activation_checkpoint must be a boolean")


def _shape(inputs: torch.Tensor, dimensions: int) -> None:
    if (not isinstance(inputs, torch.Tensor) or inputs.ndim != dimensions
            or inputs.shape[0] < 1 or inputs.shape[-3] != 2
            or inputs.shape[-2] != inputs.shape[-1] or inputs.shape[-1] < 3
            or (dimensions == 5 and inputs.shape[1] < 1)):
        shape = "(B, 2, L, L)" if dimensions == 4 else "(B, S, 2, L, L)"
        raise ValueError(f"Snapshot inputs must have nonempty shape {shape}, L >= 3")
    if inputs.is_complex() or inputs.dtype == torch.bool:
        raise ValueError("Snapshot inputs must be real numeric Z values")


class _SnapshotScaffold(nn.Module):
    """Identical named parameters and initialization in the two treatments."""

    def __init__(self, *, shot_chunk_size: int, activation_checkpoint: bool) -> None:
        super().__init__()
        self.shot_chunk_size = shot_chunk_size
        self.activation_checkpoint = activation_checkpoint
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, padding_mode="circular"), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, padding_mode="circular"), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1, padding_mode="circular"), nn.ReLU())
        # A nonlinear rho is shared across both treatments. A linear head after
        # pooling alone would reduce the set model to an additive shot predictor.
        self.head = nn.Sequential(nn.Conv2d(32, 32, 1), nn.ReLU(), nn.Conv2d(32, 2, 1))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)


class MeanSnapshotCNN(_SnapshotScaffold):
    """Apply shared-shaped nonlinear convolutions to already averaged Z maps.

    The caller averages the SAME selected shots used by SetSnapshotCNN, without
    sampling or fitting extra data. Input is (B,2,L,L); output is (B,2,L,L).
    """

    def __init__(self, *, shot_chunk_size: int = 32, activation_checkpoint: bool = True) -> None:
        _options("mean_cnn", shot_chunk_size, activation_checkpoint)
        super().__init__(shot_chunk_size=shot_chunk_size, activation_checkpoint=activation_checkpoint)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _shape(inputs, 4)
        return self.head(self.encoder(inputs.to(dtype=self.head[0].weight.dtype)))


class SetSnapshotCNN(_SnapshotScaffold):
    """Shared per-shot convolutions, spatially aligned mean pooling, then head.

    Shot microchunks bound hidden-activation memory. Non-reentrant activation
    checkpointing recomputes each encoder chunk in backward rather than retaining
    its internal activations. There are no stochastic or running-stat layers.
    Pooling is permutation invariant mathematically; float32 summation can differ
    slightly with chunk boundaries or shot ordering. Chunk size is bound during
    exact resume. No attention or shot positions are used.
    """

    def __init__(self, *, shot_chunk_size: int = 32, activation_checkpoint: bool = True) -> None:
        _options("set_cnn", shot_chunk_size, activation_checkpoint)
        super().__init__(shot_chunk_size=shot_chunk_size, activation_checkpoint=activation_checkpoint)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _shape(inputs, 5)
        batch, shots, _, height, width = inputs.shape
        pooled = None
        for start in range(0, shots, self.shot_chunk_size):
            stop = min(start + self.shot_chunk_size, shots)
            chunk = inputs[:, start:stop].to(dtype=self.head[0].weight.dtype)
            flat = chunk.reshape(batch * (stop - start), 2, height, width)
            if self.activation_checkpoint and self.training and torch.is_grad_enabled():
                encoded = checkpoint(self.encoder, flat, use_reentrant=False, preserve_rng_state=True)
            else:
                encoded = self.encoder(flat)
            contribution = encoded.reshape(batch, stop - start, 32, height, width).sum(dim=1)
            pooled = contribution if pooled is None else pooled + contribution
        return self.head(pooled / shots)


def make_snapshot_model(method: str, *, shot_chunk_size: int = 32,
                        activation_checkpoint: bool = True) -> nn.Module:
    """Construct one registered model; construction performs no CUDA operation."""
    _options(method, shot_chunk_size, activation_checkpoint)
    cls = MeanSnapshotCNN if method == "mean_cnn" else SetSnapshotCNN
    return cls(shot_chunk_size=shot_chunk_size, activation_checkpoint=activation_checkpoint)


def snapshot_model_spec(method: str, *, shot_chunk_size: int = 32,
                        activation_checkpoint: bool = True) -> dict:
    """Owned architecture plus exact-computation binding; does not touch RNG."""
    _options(method, shot_chunk_size, activation_checkpoint)
    return deepcopy({
        "schema_version": 1, "method": method, "parameter_count": 38690,
        "input_channels": ["horizontal_edge_Z", "vertical_edge_Z"],
        "input_layout": "B,2,L,L" if method == "mean_cnn" else "B,S,2,L,L",
        "output_channels": ["horizontal_magnitude", "vertical_magnitude"],
        "hidden_channels": [32, 64, 32], "hidden_kernel_size": 3,
        "hidden_padding": "circular", "hidden_activation": "ReLU",
        "head_kernel_size": 1, "head_hidden_channels": [32],
        "head_hidden_activation": "ReLU", "head_activation": "linear",
        "convolution_bias": True, "convolution_weight_initialization": "xavier_uniform",
        "convolution_bias_initialization": "zeros", "normalization": None,
        "residual_connection": False, "convolutional_path_receptive_field": 7,
        "pooling": "mean Z before encoder" if method == "mean_cnn" else "mean encoded features over shots before head",
        "shot_order_used": False, "clip_inputs": False, "clip_outputs": False,
        "shot_chunk_size": shot_chunk_size, "activation_checkpoint": activation_checkpoint,
    })
