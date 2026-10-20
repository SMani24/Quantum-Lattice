"""Fixed, prospective CNN variants for an observable-map architecture screen.

The original :class:`PeriodicObservableCNN` remains unchanged and is returned
directly for ``baseline``. All variants consume ``(batch, 3, L, L)`` observable
maps and return raw ``(batch, 2, L, L)`` edge-magnitude regressions. They use
three circular 3x3 convolutions followed by a linear 1x1 head. This convolutional
path spans a nominal 7x7 patch before torus wraparound. Normalization can add
global dependence: GroupNorm uses the entire torus in training and evaluation;
BatchNorm uses the batch and torus during training, then fixed running statistics
during evaluation.

Normalization changes the parameter count slightly. The residual arm adds a
parameter-free skip while preserving the baseline convolution widths. No variant guarantees an
improvement, positivity, rotation equivariance, or independent-shot access.
"""

from __future__ import annotations

from copy import deepcopy

from torch import nn

from quantum_lattice.models.observable_cnn import (
    PeriodicObservableCNN,
    _validate_maps,
)


VARIANT_NAMES = ("baseline", "batchnorm", "groupnorm", "residual")


def _name(name: str) -> str:
    if not isinstance(name, str) or name not in VARIANT_NAMES:
        raise ValueError(f"Unknown CNN variant {name!r}; choose from {VARIANT_NAMES}")
    return name


def _initialize(module: nn.Module) -> None:
    """Use the baseline's convolution initialization and identity norm affine."""
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)


class NormalizedObservableCNN(nn.Module):
    """Baseline convolutions with a fixed BatchNorm or GroupNorm treatment.

    Each hidden block is Conv -> Norm -> ReLU. Conv biases are intentionally
    retained to avoid changing the base convolution parameterization. BatchNorm
    maintains running means/variances and counters; its complete state must be
    saved and evaluated with ``model.eval()``. GroupNorm has no running buffers.
    """

    def __init__(self, normalization: str) -> None:
        super().__init__()
        if normalization not in ("batchnorm", "groupnorm"):
            raise ValueError("normalization must be 'batchnorm' or 'groupnorm'")
        self.normalization = normalization
        self.hidden_channels = (32, 64, 32)
        self.kernel_size = 3
        layers: list[nn.Module] = []
        previous = 3
        for width in self.hidden_channels:
            layers.append(nn.Conv2d(previous, width, 3, padding=1, padding_mode="circular"))
            if normalization == "batchnorm":
                layers.append(nn.BatchNorm2d(width, eps=1e-5, momentum=0.1,
                                             affine=True, track_running_stats=True))
            else:
                layers.append(nn.GroupNorm(8, width, eps=1e-5, affine=True))
            layers.append(nn.ReLU())
            previous = width
        layers.append(nn.Conv2d(previous, 2, 1))
        self.layers = nn.Sequential(*layers)
        _initialize(self)

    def forward(self, inputs):
        _validate_maps(inputs)
        return self.layers(inputs)


class ResidualObservableCNN(nn.Module):
    """Baseline convolutions with a parameter-free two-convolution skip.

    ``h = ReLU(stem(x))`` and ``r = conv2(ReLU(conv1(h)))`` produce
    ``head(ReLU(h + r))``. The block expands 32 -> 64 -> 32 and needs no
    projection shortcut. There is no normalization. Its 37,922 parameters and
    convolution shapes exactly match the baseline.
    """

    def __init__(self) -> None:
        super().__init__()
        self.hidden_channels = (32, 64, 32)
        self.kernel_size = 3
        self.stem = nn.Conv2d(3, 32, 3, padding=1, padding_mode="circular")
        self.block_conv1 = nn.Conv2d(32, 64, 3, padding=1, padding_mode="circular")
        self.block_conv2 = nn.Conv2d(64, 32, 3, padding=1, padding_mode="circular")
        self.activation = nn.ReLU()
        self.head = nn.Conv2d(32, 2, 1)
        _initialize(self)

    def forward(self, inputs):
        _validate_maps(inputs)
        hidden = self.activation(self.stem(inputs))
        residual = self.block_conv2(self.activation(self.block_conv1(hidden)))
        return self.head(self.activation(hidden + residual))


def make_model(name: str) -> nn.Module:
    """Construct exactly one registered model on the default Torch device."""
    name = _name(name)
    if name == "baseline":
        return PeriodicObservableCNN()
    if name == "residual":
        return ResidualObservableCNN()
    return NormalizedObservableCNN(name)


def model_spec(name: str) -> dict:
    """Return an independent JSON-safe specification without touching any RNG.

    Counts describe trainable parameters, excluding BatchNorm running buffers.
    The explicit defaults are part of the prospective experimental treatment.
    """
    name = _name(name)
    spec = {
        "schema_version": 1,
        "variant": name,
        "input_channels": ["A", "AA_horizontal", "AA_vertical"],
        "output_channels": ["horizontal_magnitude", "vertical_magnitude"],
        "hidden_channels": [32, 64, 32],
        "hidden_kernel_size": 3,
        "hidden_padding": "circular",
        "hidden_activation": "ReLU",
        "head_kernel_size": 1,
        "head_activation": "linear",
        "convolution_bias": True,
        "convolution_weight_initialization": "xavier_uniform",
        "convolution_bias_initialization": "zeros",
        "normalization": None,
        "residual_connection": False,
        "convolutional_path_receptive_field": 7,
        "effective_context": {
            "training": "7x7 convolutional patch with periodic wraparound",
            "evaluation": "7x7 convolutional patch with periodic wraparound",
        },
        "parameter_count": 37922,
        "clip_inputs": False,
        "clip_outputs": False,
    }
    if name in ("batchnorm", "groupnorm"):
        spec["normalization"] = {
            "type": "BatchNorm2d" if name == "batchnorm" else "GroupNorm",
            "placement": "after_hidden_convolution_before_relu",
            "eps": 1e-5,
            "affine": True,
            "weight_initialization": "ones",
            "bias_initialization": "zeros",
        }
        if name == "batchnorm":
            spec["normalization"].update(momentum=0.1, track_running_stats=True)
            spec["effective_context"]["training"] = "whole minibatch and torus via BatchNorm statistics"
            spec["effective_context"]["evaluation"] = (
                "7x7 convolutional patch with periodic wraparound; fixed running statistics")
        else:
            spec["normalization"]["num_groups"] = 8
            spec["effective_context"] = {
                "training": "whole torus within each channel group",
                "evaluation": "whole torus within each channel group",
            }
        spec["parameter_count"] = 38178
    elif name == "residual":
        spec["residual_connection"] = True
        spec["residual_block"] = "ReLU(h + conv2(ReLU(conv1(h))))"
    return deepcopy(spec)
