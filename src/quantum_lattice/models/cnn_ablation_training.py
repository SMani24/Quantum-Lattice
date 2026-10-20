"""Checkpointable CNN variants using the frozen whole-Hamiltonian engine.

Only model construction and the checkpoint envelope differ from
``ObservableTrainSession``. Its updates, strict-best selection, stopping replay,
Adam validation, data fingerprint and shuffle RNG remain inherited unchanged.
The explicit envelope prevents a checkpoint from silently changing architecture.
This module opens no datasets and performs no experiment scheduling.
"""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any

import numpy as np
import torch

from .cnn_variants import make_model, model_spec
from .mlp_training import resolve_device
from .observable_training import ObservableTrainSession, _array, _config


class CnnAblationTrainSession(ObservableTrainSession):
    """Whole-Hamiltonian training with an independently bound CNN variant.

    ``config`` retains the frozen engine's optimizer/stopping schema. Its
    architecture entry must be the common [32, 64, 32], 3x3 scaffold; the actual
    variant, normalization and residual details are bound separately by
    ``variant`` and :func:`model_spec`. Baseline initialization and updates are
    exactly those of the frozen periodic CNN under an equal seed and policy.

    A saved checkpoint contains an explicit architecture envelope and an
    unchanged inner engine snapshot. BatchNorm's running statistics and integer
    update counters are part of both live and selected state dictionaries.
    Validation/inference use eval mode and never update those buffers.
    """

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray,
                 val_x: np.ndarray, val_y: np.ndarray, config: dict[str, Any], *,
                 variant: str, checkpoint: dict[str, Any] | None = None) -> None:
        self.config = _config(config)
        if self.config["model"] != "cnn" or self.config["architecture"] != {
                "hidden_channels": [32, 64, 32], "kernel_size": 3}:
            raise ValueError("CNN ablations require the common cnn [32, 64, 32], kernel_size=3 scaffold")
        self.variant = variant
        self._model_spec = copy.deepcopy(model_spec(variant))
        engine_checkpoint = self._unwrap(checkpoint) if checkpoint is not None else None
        # The initialization below mirrors ObservableTrainSession.__init__;
        # make_model is the only construction difference. Calling its constructor
        # first would initialize/select the wrong model and consume extra RNG.
        self.device = resolve_device(self.config["device"], self.config["deterministic"])
        arrays = (_array(train_x, 3, "train_x"), _array(train_y, 2, "train_y"),
                  _array(val_x, 3, "val_x"), _array(val_y, 2, "val_y"))
        if arrays[0].shape[0] != arrays[1].shape[0] or arrays[2].shape[0] != arrays[3].shape[0]:
            raise ValueError("Input and target Hamiltonian counts differ")
        if len({array.shape[2:] for array in arrays}) != 1:
            raise ValueError("Training and validation lattice shapes differ")
        self.L = arrays[0].shape[2]
        digest = hashlib.sha256()
        for array in arrays:
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        self.data_fingerprint = digest.hexdigest()
        self.train_x, self.train_y, self.validation_x, self.validation_y = (
            torch.from_numpy(array.copy()).to(self.device) for array in arrays)
        torch.set_num_threads(self.config["torch_threads"])
        torch.use_deterministic_algorithms(self.config["deterministic"])
        torch.random.default_generator.manual_seed(self.config["seed"])
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config["seed"])
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = self.config["deterministic"]
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        self.model = make_model(variant).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"],
                                          betas=(0.9, 0.999), eps=1e-8)
        self.shuffle_generator = torch.Generator(device="cpu").manual_seed(self.config["seed"])
        self.permutation = torch.empty(0, dtype=torch.int64)
        self._device_permutation = self.permutation.to(self.device)
        self.cursor = self.step = self.epoch = 0
        self._epoch_squared_error = 0.0
        self._epoch_elements = 0
        self.hamiltonians_seen = 0
        self.best_step = self.best_epoch = 0
        self.best_val_mse = math.inf
        self._best_state_dict: dict[str, torch.Tensor] | None = None
        self._history: list[dict[str, Any]] = []
        self.stopped_early = False
        self.early_stopping_wait = 0
        self.early_stopping_reference: float | None = None
        if engine_checkpoint is None:
            self._validate(None)
        else:
            self._check_normalization_state(engine_checkpoint)
            super()._restore(engine_checkpoint)

    @property
    def model_spec(self) -> dict[str, Any]:
        """Return owned, versioned architecture metadata."""
        return copy.deepcopy(self._model_spec)

    def _unwrap(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        if (not isinstance(checkpoint, dict)
                or set(checkpoint) != {"training_kind", "training_schema_version", "variant", "model_spec", "engine"}
                or checkpoint.get("training_kind") != "cnn_ablation"
                or type(checkpoint.get("training_schema_version")) is not int
                or checkpoint["training_schema_version"] != 1):
            raise ValueError("Unsupported CNN-ablation checkpoint envelope")
        if checkpoint["variant"] != self.variant or checkpoint["model_spec"] != self._model_spec:
            raise ValueError("Checkpoint CNN variant/model specification differs")
        if not isinstance(checkpoint["engine"], dict):
            raise ValueError("Checkpoint inner training state must be a dictionary")
        return checkpoint["engine"]

    def _check_normalization_state(self, state: dict[str, Any]) -> None:
        """Check semantics not covered by the frozen engine's tensor checks."""
        batch_norms = [(name, module) for name, module in self.model.named_modules()
                       if isinstance(module, torch.nn.BatchNorm2d)
                       and module.track_running_stats]
        for key, counter_key in (("model_state_dict", "step"), ("best_state_dict", "best_step")):
            supplied = state.get(key)
            counter = state.get(counter_key)
            if not isinstance(supplied, dict) or type(counter) is not int or counter < 0:
                raise ValueError("Checkpoint normalization state/progress is invalid")
            for name, _ in batch_norms:
                prefix = name + "." if name else ""
                tracked = supplied.get(prefix + "num_batches_tracked")
                variance = supplied.get(prefix + "running_var")
                if (not isinstance(tracked, torch.Tensor) or tracked.dtype != torch.int64
                        or tracked.ndim != 0 or tracked.item() != counter):
                    raise ValueError("Checkpoint BatchNorm update counter differs from optimizer progress")
                if (not isinstance(variance, torch.Tensor)
                        or not bool(torch.isfinite(variance).all().item())
                        or bool(torch.any(variance < 0).item())):
                    raise ValueError("Checkpoint BatchNorm running variance is invalid")

    def snapshot(self) -> dict[str, Any]:
        """Return the variant binding and fully owned frozen-engine state."""
        engine = super().snapshot()
        self._check_normalization_state(engine)
        return {"training_kind": "cnn_ablation", "training_schema_version": 1,
                "variant": self.variant, "model_spec": self.model_spec, "engine": engine}
