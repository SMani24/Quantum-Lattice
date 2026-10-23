"""Parameter-matched plain/residual versions of the frozen raw-data CNNs.

Only the encoder's third-convolution residual addition changes. Reusing the
existing layers preserves parameter names, initialization and plain outputs;
neither factory changes the original implementation or its saved checkpoints.
Local vertex-gauge invariance is not imposed by either architecture.
"""

from __future__ import annotations

import torch
from torch import nn

from quantum_lattice.models.snapshot_cnn import (
    METHODS, make_snapshot_model, snapshot_model_spec)


STRUCTURE_SCHEMA_VERSION = 1


class _ResidualEncoder(nn.Sequential):
    """The existing named layers, with a parameter-free identity shortcut."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first = self[1](self[0](inputs))
        middle = self[3](self[2](first))
        return self[5](self[4](middle) + first)


def _residual_option(residual: bool) -> None:
    if type(residual) is not bool:
        raise ValueError("residual must be a boolean")


def make_structure_model(method: str, *, residual: bool,
                         shot_chunk_size: int = 32,
                         activation_checkpoint: bool = True) -> nn.Module:
    """Build one matched treatment without extra initialization RNG draws.

    ``residual=False`` returns the frozen plain architecture. For a residual
    treatment, h=ReLU(C1 x) and phi(x)=ReLU(C3 ReLU(C2 h)+h). Mean/set pooling,
    nonlinear readout, shot chunks and activation recomputation stay unchanged.
    """
    _residual_option(residual)
    model = make_snapshot_model(method, shot_chunk_size=shot_chunk_size,
                                activation_checkpoint=activation_checkpoint)
    if residual:
        model.encoder = _ResidualEncoder(*model.encoder.children())
    return model


def structure_model_spec(method: str, *, residual: bool,
                         shot_chunk_size: int = 32,
                         activation_checkpoint: bool = True) -> dict:
    """Return owned architecture metadata, without constructing random weights."""
    _residual_option(residual)
    spec = snapshot_model_spec(method, shot_chunk_size=shot_chunk_size,
                               activation_checkpoint=activation_checkpoint)
    spec.update({
        "structure_schema_version": STRUCTURE_SCHEMA_VERSION,
        "residual_connection": residual,
        "residual_addition": (
            "first encoder ReLU output added after third convolution before ReLU"
            if residual else None),
        "local_vertex_gauge_invariance_enforced": False,
    })
    return spec
