"""Checkpointable raw-snapshot and complete-edge-mean magnitude regression.

The immutable ObservableTrainSession supplies bounded updates, epoch0/strict-best
selection, separate min_delta stopping, Adam validation, and exact partial-epoch
resume. Only model construction, input validation, and the checkpoint envelope
differ. Its internal common CNN scaffold metadata describes hidden widths; the
outer snapshot specification explicitly binds two input channels and the method.
This module opens no files and never receives final-test data.
"""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any

import numpy as np
import torch

from .mlp_training import resolve_device
from .observable_training import ObservableTrainSession, _array, _config
from .snapshot_cnn import make_snapshot_model, snapshot_model_spec


def _settings(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    allowed = {"seed", "device", "deterministic", "torch_threads", "learning_rate",
               "batch_size_hamiltonians", "max_epochs", "validation_batch_size",
               "early_stopping", "shot_chunk_size", "activation_checkpoint"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Unknown snapshot training settings")
    execution = {"shot_chunk_size": config.get("shot_chunk_size", 32),
                 "activation_checkpoint": config.get("activation_checkpoint", True)}
    if type(execution["shot_chunk_size"]) is not int or execution["shot_chunk_size"] < 1:
        raise ValueError("shot_chunk_size must be a positive integer")
    if type(execution["activation_checkpoint"]) is not bool:
        raise ValueError("activation_checkpoint must be a boolean")
    core = {k: v for k, v in config.items() if k not in execution}
    stopping = core.pop("early_stopping", {})
    if not isinstance(stopping, dict):
        raise ValueError("early_stopping must be a dictionary")
    stopping = {"enabled": True, "min_epochs": 30, "patience_epochs": 30,
                "min_delta": 1e-6, **stopping}
    core = {"model": "cnn", "architecture": {"hidden_channels": [32, 64, 32], "kernel_size": 3},
            "device": "cuda", "learning_rate": 1e-4, "batch_size_hamiltonians": 16,
            "max_epochs": 200, "validation_batch_size": 16,
            "early_stopping": stopping, **core}
    return _config(core), execution


def _inputs(values: np.ndarray, method: str, name: str) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{name} must be real numeric Z values")
    dimensions = 4 if method == "mean_cnn" else 5
    if (array.ndim != dimensions or array.shape[0] < 1 or array.shape[-3] != 2
            or array.shape[-2] != array.shape[-1] or array.shape[-1] < 3
            or (dimensions == 5 and array.shape[1] < 1)):
        raise ValueError(f"{name} must have nonempty {dimensions}D shape with two channels and L >= 3")
    if not np.all(np.isfinite(array)) or np.any(array < -1) or np.any(array > 1):
        raise ValueError(f"{name} must contain finite Z values in [-1,1]")
    if method == "set_cnn" and np.any((array != -1) & (array != 1)):
        raise ValueError(f"{name} raw Z snapshots must contain only -1 or +1")
    return np.ascontiguousarray(array, dtype=np.float32)


class SnapshotTrainSession(ObservableTrainSession):
    """Bounded whole-Hamiltonian Adam session for one declared snapshot method.

    Public settings are seed/device/deterministic/torch_threads/learning_rate,
    batch_size_hamiltonians/max_epochs/validation_batch_size/early_stopping,
    shot_chunk_size/activation_checkpoint. CUDA is the default with no fallback;
    explicit device='cpu' supports synthetic verification only. Batches contain
    complete Hamiltonians, never independent labels per correlated shot.

    Inputs are mean maps (H,2,L,L) or raw shots (H,S,2,L,L), respectively. Targets
    always have (H,2,L,L). Train and validation must use the same shot count in the
    raw method. Arrays become owned float32 device tensors; checkpointing bounds
    hidden activation memory, not storage of the input bank. The campaign owns
    matched prefixes, Hamiltonian splits, file I/O and inference provenance.
    """

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray,
                 val_x: np.ndarray, val_y: np.ndarray, config: dict[str, Any], *,
                 method: str, checkpoint: dict[str, Any] | None = None) -> None:
        self.config, self.execution = _settings(config)
        self.method = method
        self._model_spec = snapshot_model_spec(method, **self.execution)
        engine_checkpoint = self._unwrap(checkpoint) if checkpoint is not None else None
        self.device = resolve_device(self.config["device"], self.config["deterministic"])
        arrays = (_inputs(train_x, method, "train_x"), _array(train_y, 2, "train_y"),
                  _inputs(val_x, method, "val_x"), _array(val_y, 2, "val_y"))
        if arrays[0].shape[0] != arrays[1].shape[0] or arrays[2].shape[0] != arrays[3].shape[0]:
            raise ValueError("Input and target Hamiltonian counts differ")
        if len({a.shape[-2:] for a in arrays}) != 1:
            raise ValueError("Training and validation lattice shapes differ")
        if method == "set_cnn" and arrays[0].shape[1] != arrays[2].shape[1]:
            raise ValueError("Training and validation shot counts differ")
        self.L = arrays[0].shape[-1]
        self.shots = arrays[0].shape[1] if method == "set_cnn" else None
        digest = hashlib.sha256()
        for array in arrays:
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        self.data_fingerprint = digest.hexdigest()
        self.train_x, self.train_y, self.validation_x, self.validation_y = (
            torch.from_numpy(a.copy()).to(self.device) for a in arrays)
        torch.set_num_threads(self.config["torch_threads"])
        torch.use_deterministic_algorithms(self.config["deterministic"])
        torch.random.default_generator.manual_seed(self.config["seed"])
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config["seed"])
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = self.config["deterministic"]
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        self.model = make_snapshot_model(method, **self.execution).to(self.device)
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
            super()._restore(engine_checkpoint)

    @property
    def model_spec(self) -> dict[str, Any]:
        return copy.deepcopy(self._model_spec)

    @property
    def settings(self) -> dict[str, Any]:
        """Owned public config without the immutable engine's scaffold keys."""
        return {**{k: copy.deepcopy(v) for k, v in self.config.items()
                   if k not in ("model", "architecture")}, **copy.deepcopy(self.execution)}

    def _unwrap(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        if (not isinstance(checkpoint, dict)
                or set(checkpoint) != {"training_kind", "training_schema_version", "method", "model_spec", "engine"}
                or checkpoint.get("training_kind") != "snapshot_cnn"
                or type(checkpoint.get("training_schema_version")) is not int
                or checkpoint["training_schema_version"] != 1):
            raise ValueError("Unsupported snapshot training checkpoint envelope")
        if checkpoint["method"] != self.method or checkpoint["model_spec"] != self._model_spec:
            raise ValueError("Checkpoint snapshot method/model/execution specification differs")
        if not isinstance(checkpoint["engine"], dict):
            raise ValueError("Checkpoint inner training state must be a dictionary")
        return checkpoint["engine"]

    def snapshot(self) -> dict[str, Any]:
        """Fully owned tensor/primitive state suitable for weights_only loading."""
        return {"training_kind": "snapshot_cnn", "training_schema_version": 1,
                "method": self.method, "model_spec": self.model_spec,
                "engine": super().snapshot()}

    def selected_predictions(self, inputs: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        values = _inputs(inputs, self.method, "prediction inputs")
        if values.shape[-1] != self.L:
            raise ValueError("Prediction lattice shape differs from training")
        if self.method == "set_cnn" and values.shape[1] != self.shots:
            raise ValueError("Prediction shot count differs from training")
        size = self.config["validation_batch_size"] if batch_size is None else batch_size
        if type(size) is not int or size < 1:
            raise ValueError("prediction batch_size must be a positive integer")
        model = copy.deepcopy(self.model)
        model.load_state_dict(self._best_state_dict)
        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(values), size):
                tensor = torch.from_numpy(values[start:start + size].copy()).to(self.device)
                prediction = model(tensor)
                if not bool(torch.isfinite(prediction).all().item()):
                    raise FloatingPointError("Nonfinite selected-model predictions")
                predictions.append(prediction.cpu().numpy().copy())
        return np.concatenate(predictions, axis=0)
