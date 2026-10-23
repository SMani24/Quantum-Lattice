"""Exact-resume training for the residual and whole-bag gauge ablations.

The immutable observable engine still owns validation, strict-best selection,
stopping and checkpoint validation. The small initializer and bounded update
loop here specialize model construction and training-only inputs. No global
factory is patched, and no old scientific source is modified.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np
import torch

from ..data.snapshot_gauge import gauge_mask_spec, vertex_gauge_masks
from .mlp_training import resolve_device
from .observable_training import ObservableTrainSession, _array, _integer
from .snapshot_structure import make_structure_model, structure_model_spec
from .snapshot_training import SnapshotTrainSession, _inputs, _settings


def _uids(values: Sequence[str], count: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("train_uids must contain one distinct string per Hamiltonian")
    try:
        result = tuple(values)
    except TypeError as error:
        raise ValueError("train_uids must be a sequence of strings") from error
    if (len(result) != count or any(not isinstance(value, str) or not value.strip()
                                    for value in result) or len(set(result)) != count):
        raise ValueError("train_uids must contain one distinct nonempty string per Hamiltonian")
    return result


class StructureTrainSession(SnapshotTrainSession):
    """Whole-Hamiltonian Adam with independently keyed training augmentation.

    ``train_uids`` binds the exact ordered identity of each input row, including
    when augmentation is off. One vertex-gauge mask is keyed by UID, model seed,
    lattice and zero-based optimizer epoch. It applies to every shot in a bag,
    or to that bag's complete edge mean. Targets, full training diagnostics,
    validation and selected-model inference remain unaugmented. Mask generation
    consumes neither neural initialization nor shuffle RNG state.

    CUDA is the default; explicit device='cpu' is for synthetic verification.
    The caller owns file I/O, split/record provenance and long-run scheduling.
    """

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray,
                 val_x: np.ndarray, val_y: np.ndarray, config: dict[str, Any], *,
                 method: str, residual: bool, gauge_augmentation: bool,
                 train_uids: Sequence[str],
                 checkpoint: dict[str, Any] | None = None) -> None:
        if type(residual) is not bool or type(gauge_augmentation) is not bool:
            raise ValueError("residual and gauge_augmentation must be booleans")
        self.config, self.execution = _settings(config)
        self.method, self.residual = method, residual
        self.gauge_augmentation = gauge_augmentation
        self._model_spec = structure_model_spec(method, residual=residual, **self.execution)
        arrays = (_inputs(train_x, method, "train_x"), _array(train_y, 2, "train_y"),
                  _inputs(val_x, method, "val_x"), _array(val_y, 2, "val_y"))
        if arrays[0].shape[0] != arrays[1].shape[0] or arrays[2].shape[0] != arrays[3].shape[0]:
            raise ValueError("Input and target Hamiltonian counts differ")
        if len({array.shape[-2:] for array in arrays}) != 1:
            raise ValueError("Training and validation lattice shapes differ")
        if method == "set_cnn" and arrays[0].shape[1] != arrays[2].shape[1]:
            raise ValueError("Training and validation shot counts differ")
        self.L = arrays[0].shape[-1]
        self.shots = arrays[0].shape[1] if method == "set_cnn" else None
        self.train_uids = _uids(train_uids, len(arrays[0]))
        encoded_uids = json.dumps(self.train_uids, ensure_ascii=True, separators=(",", ":")).encode("utf8")
        self._augmentation_spec = {
            "mask_spec": gauge_mask_spec(), "enabled": gauge_augmentation,
            "seed": self.config["seed"], "epoch_index": "zero_based_optimizer_epoch",
            "scope": "whole_hamiltonian_bag", "validation": "unaugmented",
            "train_uids_sha256": hashlib.sha256(encoded_uids).hexdigest()}
        self._mask_epoch: int | None = None
        self._epoch_masks: torch.Tensor | None = None
        engine_checkpoint = self._unwrap(checkpoint) if checkpoint is not None else None
        self.device = resolve_device(self.config["device"], self.config["deterministic"])
        digest = hashlib.sha256(encoded_uids)
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
        self.model = make_structure_model(method, residual=residual, **self.execution).to(self.device)
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
            ObservableTrainSession._restore(self, engine_checkpoint)

    @property
    def augmentation_spec(self) -> dict[str, Any]:
        return copy.deepcopy(self._augmentation_spec)

    def _unwrap(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        keys = {"training_kind", "training_schema_version", "method", "model_spec",
                "residual", "gauge_augmentation", "augmentation_spec", "train_uids", "engine"}
        if (not isinstance(checkpoint, dict) or set(checkpoint) != keys
                or checkpoint.get("training_kind") != "snapshot_structure_cnn"
                or type(checkpoint.get("training_schema_version")) is not int
                or checkpoint["training_schema_version"] != 1):
            raise ValueError("Unsupported structural training checkpoint envelope")
        if (checkpoint["method"] != self.method or checkpoint["model_spec"] != self._model_spec
                or type(checkpoint["residual"]) is not bool or checkpoint["residual"] != self.residual
                or type(checkpoint["gauge_augmentation"]) is not bool
                or checkpoint["gauge_augmentation"] != self.gauge_augmentation
                or checkpoint["augmentation_spec"] != self._augmentation_spec
                or checkpoint["train_uids"] != list(self.train_uids)):
            raise ValueError("Checkpoint structural factors, augmentation or Hamiltonian UIDs differ")
        if not isinstance(checkpoint["engine"], dict):
            raise ValueError("Checkpoint inner training state must be a dictionary")
        return checkpoint["engine"]

    def snapshot(self) -> dict[str, Any]:
        """Owned weights-only-compatible state; epoch-keyed masks regenerate."""
        return {"training_kind": "snapshot_structure_cnn", "training_schema_version": 1,
                "method": self.method, "model_spec": self.model_spec,
                "residual": self.residual, "gauge_augmentation": self.gauge_augmentation,
                "augmentation_spec": self.augmentation_spec, "train_uids": list(self.train_uids),
                "engine": ObservableTrainSession.snapshot(self)}

    def _training_inputs(self, indices: torch.Tensor) -> torch.Tensor:
        inputs = self.train_x[indices]
        if self.gauge_augmentation:
            if self._mask_epoch != self.epoch:
                masks = vertex_gauge_masks(self.train_uids, self.L,
                                           seed=self.config["seed"], epoch=self.epoch)
                self._epoch_masks = torch.from_numpy(masks).to(device=self.device, dtype=inputs.dtype)
                self._mask_epoch = self.epoch
            masks = self._epoch_masks[indices]
            if self.method == "set_cnn":
                masks = masks[:, None]
            inputs = inputs * masks
        return inputs

    def advance(self, max_updates: int) -> list[dict[str, Any]]:
        """Run a bounded number of Adam updates, augmenting only training inputs."""
        _integer(max_updates, "max_updates", 0)
        events = []
        for _ in range(max_updates):
            if self.finished:
                break
            if self.cursor >= len(self.permutation):
                self.permutation = torch.randperm(len(self.train_x), generator=self.shuffle_generator)
                self._device_permutation = self.permutation.to(self.device)
                self.cursor = 0
            stop = min(self.cursor + self.config["batch_size_hamiltonians"], len(self.train_x))
            indices = self._device_permutation[self.cursor:stop]
            inputs, targets = self._training_inputs(indices), self.train_y[indices]
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            loss = (self.model(inputs) - targets).square().mean()
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            gradients = [torch.isfinite(parameter.grad).all() for parameter in self.model.parameters()
                         if parameter.grad is not None]
            if not gradients or not bool(torch.stack(gradients).all().item()):
                raise FloatingPointError("Nonfinite or absent training gradients")
            self.optimizer.step()
            self.cursor = stop
            self.step += 1
            self.hamiltonians_seen += len(inputs)
            self._epoch_squared_error += float(loss.detach().item()) * targets.numel()
            self._epoch_elements += targets.numel()
            if self.cursor == len(self.train_x):
                self.epoch += 1
                optimizer_epoch_mse = self._epoch_squared_error / self._epoch_elements
                self._epoch_squared_error, self._epoch_elements = 0.0, 0
                events.append(self._validate(optimizer_epoch_mse))
        return events
