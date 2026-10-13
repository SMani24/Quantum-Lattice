"""Checkpointable training with validation-only model selection.

The training engine never receives final-test data. A snapshot contains only
tensors and primitive containers, so its caller can load it using PyTorch's
``weights_only=True`` restriction. Disk writes belong to the experiment runner.
"""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any

import numpy as np
import torch

from .mlp import PaperMLP


def _cpu_copy(value: Any) -> Any:
    """Detach checkpoint tensors without retaining references to live state."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _array(value: np.ndarray, columns: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != columns or array.shape[0] == 0:
        raise ValueError(f"{name} must have nonempty shape (rows, {columns})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return np.ascontiguousarray(array)


def _config(settings: dict[str, Any]) -> dict[str, Any]:
    config = {
        "seed": settings.get("seed", 0),
        "learning_rate": settings.get("learning_rate", 1e-5),
        "batch_size": settings.get("batch_size", 128),
        "max_steps": settings.get("max_steps", 10000),
        "validation_interval": settings.get("validation_interval", 100),
        "device": settings.get("device", "auto"),
        "torch_threads": settings.get("torch_threads", 1),
        "deterministic": settings.get("deterministic", True),
    }
    for key in ("seed", "batch_size", "max_steps", "validation_interval", "torch_threads"):
        if isinstance(config[key], bool) or not isinstance(config[key], int):
            raise ValueError(f"{key} must be an integer")
        if config[key] < (0 if key == "seed" else 1):
            raise ValueError(f"{key} is out of range")
    if config["seed"] >= 2**63:
        raise ValueError("seed must be smaller than 2**63")
    if isinstance(config["learning_rate"], bool):
        raise ValueError("learning_rate must be finite and positive")
    config["learning_rate"] = float(config["learning_rate"])
    if not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if config["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if not isinstance(config["deterministic"], bool):
        raise ValueError("deterministic must be a boolean")
    return config


class TrainSession:
    """A resumable Adam session, stepped in bounded chunks by the runner.

    ``step`` counts optimizer updates. ``epoch`` counts shuffled passes started;
    a pass includes its final, possibly smaller, batch. Validation is performed
    after the first update, at each interval and at the final configured update.
    """

    def __init__(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        validation_x: np.ndarray,
        validation_y: np.ndarray,
        config: dict[str, Any],
        checkpoint: dict[str, Any] | None = None,
    ) -> None:
        self.config = _config(config)
        arrays = (
            _array(train_x, 3, "train_x"),
            _array(train_y, 1, "train_y"),
            _array(validation_x, 3, "validation_x"),
            _array(validation_y, 1, "validation_y"),
        )
        if arrays[0].shape[0] != arrays[1].shape[0]:
            raise ValueError("Training input and target counts differ")
        if arrays[2].shape[0] != arrays[3].shape[0]:
            raise ValueError("Validation input and target counts differ")
        digest = hashlib.sha256()
        for array in arrays:
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        self.data_fingerprint = digest.hexdigest()
        # Copies prevent a caller's subsequent array edits from changing a run.
        self.train_x, self.train_y, self.validation_x, self.validation_y = (
            torch.from_numpy(array.copy()) for array in arrays
        )

        requested = self.config["device"]
        if requested == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        actual = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
        self.device = torch.device("cpu" if actual == "auto" else actual)
        torch.set_num_threads(self.config["torch_threads"])
        torch.use_deterministic_algorithms(self.config["deterministic"])
        torch.manual_seed(self.config["seed"])
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config["seed"])
        self.model = PaperMLP().to(self.device)
        # Explicit defaults expose the optimizer adaptation from legacy TF1.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"],
            betas=(0.9, 0.999), eps=1e-8,
        )
        self.shuffle_generator = torch.Generator(device="cpu")
        self.shuffle_generator.manual_seed(self.config["seed"])
        self.permutation = torch.empty(0, dtype=torch.int64)
        self.cursor = 0
        self.step = 0
        self.epoch = 0
        self.best_val_mse = math.inf
        self.best_step: int | None = None
        self._best_state_dict: dict[str, torch.Tensor] | None = None
        self.last_validation: dict[str, Any] | None = None
        if checkpoint is not None:
            self._restore(checkpoint)

    @property
    def finished(self) -> bool:
        return self.step >= self.config["max_steps"]

    @property
    def best_state_dict(self) -> dict[str, torch.Tensor] | None:
        return _cpu_copy(self._best_state_dict)

    def _batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cursor >= len(self.permutation):
            self.permutation = torch.randperm(
                len(self.train_x), generator=self.shuffle_generator
            )
            self.cursor = 0
            self.epoch += 1
        stop = min(self.cursor + self.config["batch_size"], len(self.train_x))
        indices = self.permutation[self.cursor:stop]
        self.cursor = stop
        return self.train_x[indices].to(self.device), self.train_y[indices].to(self.device)

    def _mse(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        squared_error = 0.0
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(inputs), 4096):
                prediction = self.model(inputs[start:start + 4096].to(self.device))
                difference = prediction - targets[start:start + 4096].to(self.device)
                squared_error += float(torch.sum(difference.double().square()).item())
        return squared_error / len(inputs)

    def _validate(self, batch_mse: float) -> dict[str, Any]:
        validation_mse = self._mse(self.validation_x, self.validation_y)
        train_mse = self._mse(self.train_x, self.train_y)
        if not math.isfinite(validation_mse) or not math.isfinite(train_mse):
            raise FloatingPointError("Nonfinite model predictions during validation")
        improved = validation_mse < self.best_val_mse
        if improved:
            self.best_val_mse = validation_mse
            self.best_step = self.step
            self._best_state_dict = _cpu_copy(self.model.state_dict())
        event = {
            "step": self.step, "epoch": self.epoch, "batch_mse": batch_mse,
            "train_mse": train_mse, "validation_mse": validation_mse,
            "best_validation_mse": self.best_val_mse, "best_step": self.best_step,
            "improved": improved,
        }
        self.last_validation = event
        return event.copy()

    def advance(self, steps: int) -> list[dict[str, Any]]:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        events = []
        for _ in range(min(steps, self.config["max_steps"] - self.step)):
            inputs, targets = self._batch()
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((self.model(inputs) - targets).square())
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                   for parameter in self.model.parameters()):
                raise FloatingPointError("Nonfinite training gradients")
            self.optimizer.step()
            self.step += 1
            if self.step == 1 or self.step % self.config["validation_interval"] == 0 or self.finished:
                events.append(self._validate(float(loss.detach().item())))
        return events

    def predict(self, inputs: np.ndarray, use_best: bool = True) -> np.ndarray:
        values = torch.from_numpy(_array(inputs, 3, "prediction inputs"))
        if use_best:
            if self._best_state_dict is None:
                raise RuntimeError("No validation-selected checkpoint exists yet")
            # Deep-copying avoids new random initialization changing session RNG.
            model = copy.deepcopy(self.model)
            model.load_state_dict(self._best_state_dict)
        else:
            model = self.model
        was_training = model.training
        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(values), 4096):
                predictions.append(model(values[start:start + 4096].to(self.device)).cpu().numpy())
        model.train(was_training)
        return np.concatenate(predictions, axis=0)

    def snapshot(self) -> dict[str, Any]:
        return {
            "training_schema_version": 1,
            "config": self.config.copy(),
            "data_fingerprint": self.data_fingerprint,
            "device_type": self.device.type,
            "model_state_dict": _cpu_copy(self.model.state_dict()),
            "optimizer_state_dict": _cpu_copy(self.optimizer.state_dict()),
            "shuffle_generator_state": self.shuffle_generator.get_state().clone(),
            "permutation": self.permutation.clone(), "cursor": self.cursor,
            "step": self.step, "epoch": self.epoch,
            "best_val_mse": self.best_val_mse, "best_step": self.best_step,
            "best_state_dict": self.best_state_dict,
            "last_validation": copy.deepcopy(self.last_validation),
            "torch_rng_state": torch.get_rng_state().clone(),
            "cuda_rng_states": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
            if self.device.type == "cuda" else [],
        }

    def _restore(self, checkpoint: dict[str, Any]) -> None:
        if checkpoint.get("training_schema_version") != 1:
            raise ValueError("Unsupported training checkpoint version")
        if checkpoint.get("data_fingerprint") != self.data_fingerprint:
            raise ValueError("Checkpoint training/validation data differ")
        # Runtime CPU thread count and explicit/auto device spelling may change.
        scientific = set(self.config) - {"device", "torch_threads"}
        if any(checkpoint["config"].get(key) != self.config[key] for key in scientific):
            raise ValueError("Checkpoint training settings differ")
        if checkpoint.get("device_type") != self.device.type:
            raise ValueError("Exact resume requires the original CPU/CUDA device type")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.shuffle_generator.set_state(checkpoint["shuffle_generator_state"].cpu())
        self.permutation = checkpoint["permutation"].cpu().clone()
        self.cursor = int(checkpoint["cursor"])
        self.step = int(checkpoint["step"])
        self.epoch = int(checkpoint["epoch"])
        self.best_val_mse = float(checkpoint["best_val_mse"])
        self.best_step = checkpoint["best_step"]
        self._best_state_dict = _cpu_copy(checkpoint["best_state_dict"])
        self.last_validation = copy.deepcopy(checkpoint["last_validation"])
        if not 0 <= self.cursor <= len(self.permutation) or not 0 <= self.step <= self.config["max_steps"]:
            raise ValueError("Invalid checkpoint progress counters")
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_states"]])
