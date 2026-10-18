"""Checkpointable whole-Hamiltonian training for observable-map regressors.

The engine receives train/validation arrays only. Each shuffled batch consists
of complete Hamiltonians, and an epoch includes its final short batch. Initial
weights and every completed epoch are eligible for validation-only selection.
The caller owns disk I/O, source/data provenance, campaign scheduling and final
evaluation; this module never opens a dataset or selects final-test records.
"""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any

import numpy as np
import torch

from .mlp_training import resolve_device
from .observable_cnn import ObservableMapMLP, PeriodicObservableCNN


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def _config(settings: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(settings, dict):
        raise ValueError("Training config must be a dictionary")
    allowed = {"model", "architecture", "seed", "device", "deterministic",
               "torch_threads", "learning_rate", "batch_size_hamiltonians",
               "max_epochs", "validation_batch_size", "early_stopping"}
    if set(settings) - allowed:
        raise ValueError(f"Unknown training settings: {sorted(set(settings) - allowed)}")
    config = {"model": settings.get("model", "cnn"),
              "architecture": copy.deepcopy(settings.get("architecture", {
                  "hidden_channels": [32, 64, 32], "kernel_size": 3})),
              "seed": settings.get("seed", 0),
              "device": settings.get("device", "cuda"),
              "deterministic": settings.get("deterministic", True),
              "torch_threads": settings.get("torch_threads", 1),
              "learning_rate": settings.get("learning_rate", 1e-4),
              "batch_size_hamiltonians": settings.get("batch_size_hamiltonians", 32),
              "max_epochs": settings.get("max_epochs", 500),
              "validation_batch_size": settings.get("validation_batch_size", 256)}
    if config["model"] not in ("cnn", "map_mlp"):
        raise ValueError("model must be cnn or map_mlp")
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda; silent fallback is not permitted")
    if not isinstance(config["deterministic"], bool):
        raise ValueError("deterministic must be a boolean")
    for name in ("torch_threads", "batch_size_hamiltonians", "max_epochs", "validation_batch_size"):
        _integer(config[name], name)
    _integer(config["seed"], "seed", 0)
    if config["seed"] >= 2**63:
        raise ValueError("seed must be smaller than 2**63")
    config["learning_rate"] = _number(config["learning_rate"], "learning_rate", positive=True)
    architecture = config["architecture"]
    if not isinstance(architecture, dict) or set(architecture) != {"hidden_channels", "kernel_size"}:
        raise ValueError("architecture requires hidden_channels and kernel_size")
    widths = architecture["hidden_channels"]
    if not isinstance(widths, (list, tuple)) or not widths:
        raise ValueError("architecture.hidden_channels must contain positive integer widths")
    for width in widths:
        _integer(width, "architecture.hidden_channels width")
    architecture["hidden_channels"] = list(widths)
    if type(architecture["kernel_size"]) is not int or architecture["kernel_size"] not in (1, 3):
        raise ValueError("architecture.kernel_size must be 1 or 3")
    stopping = settings.get("early_stopping", {})
    if not isinstance(stopping, dict) or set(stopping) - {"enabled", "min_epochs", "patience_epochs", "min_delta"}:
        raise ValueError("Invalid early_stopping settings")
    config["early_stopping"] = {
        "enabled": stopping.get("enabled", True),
        "min_epochs": stopping.get("min_epochs", 50),
        "patience_epochs": stopping.get("patience_epochs", 30),
        "min_delta": stopping.get("min_delta", 1e-6)}
    stopping = config["early_stopping"]
    if not isinstance(stopping["enabled"], bool):
        raise ValueError("early_stopping.enabled must be a boolean")
    _integer(stopping["min_epochs"], "early_stopping.min_epochs", 0)
    _integer(stopping["patience_epochs"], "early_stopping.patience_epochs")
    stopping["min_delta"] = _number(stopping["min_delta"], "early_stopping.min_delta", positive=False)
    return config


def _array(value: np.ndarray, channels: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{name} must be real numeric values")
    if (array.ndim != 4 or array.shape[0] < 1 or array.shape[1] != channels
            or array.shape[2] < 3 or array.shape[2] != array.shape[3]):
        raise ValueError(f"{name} must have nonempty shape (N, {channels}, L, L), L >= 3")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.ascontiguousarray(array, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} exceeds finite float32 range")
    if channels == 2 and np.any(result < 0):
        raise ValueError(f"{name} magnitude targets must be nonnegative")
    return result


class ObservableTrainSession:
    """Bounded Adam updates with exact-resume state and epoch-based selection.

    The independent CPU shuffle generator gives the same Hamiltonian order for
    equal counts/seeds regardless of architecture or estimator values. Raw
    outputs are linear and unclipped. Full-data MSE averages over Hamiltonians,
    both edge orientations and every lattice cell; validation runs at epoch 0
    and after every completed shuffled pass. Best weights use strict loss
    improvement, while stopping uses a separate absolute ``min_delta``.
    """

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray,
                 val_x: np.ndarray, val_y: np.ndarray, config: dict[str, Any],
                 checkpoint: dict[str, Any] | None = None) -> None:
        self.config = _config(config)
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
        # Avoid touching CUDA generators at all for the CPU control path.
        torch.random.default_generator.manual_seed(self.config["seed"])
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config["seed"])
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = self.config["deterministic"]
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        self.model = (PeriodicObservableCNN(**self.config["architecture"])
                      if self.config["model"] == "cnn" else ObservableMapMLP()).to(self.device)
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
        if checkpoint is None:
            self._validate(None)
        else:
            self._restore(checkpoint)

    @property
    def finished(self) -> bool:
        return self.stopped_early or self.epoch >= self.config["max_epochs"]

    @property
    def stop_reason(self) -> str | None:
        if self.stopped_early:
            return "early_stopping"
        return "max_epochs" if self.epoch >= self.config["max_epochs"] else None

    @property
    def history(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._history)

    @property
    def best_state_dict(self) -> dict[str, torch.Tensor]:
        return _cpu_copy(self._best_state_dict)

    def _mse(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        total = torch.zeros((), dtype=torch.float64, device=self.device)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for start in range(0, len(inputs), self.config["validation_batch_size"]):
                    stop = start + self.config["validation_batch_size"]
                    difference = self.model(inputs[start:stop]) - targets[start:stop]
                    total += difference.double().square().sum()
        finally:
            self.model.train(was_training)
        return float(total.item()) / targets.numel()

    def _validate(self, optimizer_epoch_mse: float | None) -> dict[str, Any]:
        train_mse = self._mse(self.train_x, self.train_y)
        val_mse = self._mse(self.validation_x, self.validation_y)
        if not math.isfinite(train_mse) or not math.isfinite(val_mse):
            raise FloatingPointError("Nonfinite full-data loss during validation")
        improved = val_mse < self.best_val_mse
        if improved:
            self.best_val_mse = val_mse
            self.best_step, self.best_epoch = self.step, self.epoch
            self._best_state_dict = _cpu_copy(self.model.state_dict())
        stopping = self.config["early_stopping"]
        if self.epoch <= stopping["min_epochs"]:
            self.early_stopping_wait = 0
            self.early_stopping_reference = self.best_val_mse
        elif stopping["enabled"]:
            if (self.early_stopping_reference is None
                    or val_mse < self.early_stopping_reference - stopping["min_delta"]):
                self.early_stopping_reference = val_mse
                self.early_stopping_wait = 0
            else:
                self.early_stopping_wait += 1
                self.stopped_early = self.early_stopping_wait >= stopping["patience_epochs"]
        event = {"step": self.step, "epoch": self.epoch,
                 "hamiltonians_seen": self.hamiltonians_seen,
                 "optimizer_epoch_mse": optimizer_epoch_mse,
                 "train_mse": train_mse, "validation_mse": val_mse,
                 "best_validation_mse": self.best_val_mse,
                 "best_step": self.best_step, "best_epoch": self.best_epoch,
                 "improved": improved, "early_stopping_wait": self.early_stopping_wait,
                 "early_stopping_reference": self.early_stopping_reference,
                 "stopped_early": self.stopped_early, "stop_reason": self.stop_reason}
        self._history.append(event)
        return copy.deepcopy(event)

    def advance(self, max_updates: int) -> list[dict[str, Any]]:
        """Run at most ``max_updates`` Adam updates; return new epoch checks."""
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
            inputs, targets = self.train_x[indices], self.train_y[indices]
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

    def selected_predictions(self, inputs: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        """Predict with selected weights without changing live state or RNG."""
        values = _array(inputs, 3, "prediction inputs")
        if values.shape[2] != self.L:
            raise ValueError("Prediction lattice shape differs from the training lattice")
        size = self.config["validation_batch_size"] if batch_size is None else batch_size
        _integer(size, "prediction batch_size")
        # deepcopy performs no random model initialization, unlike a constructor.
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

    def snapshot(self) -> dict[str, Any]:
        """Return owned primitive/tensor state suitable for weights-only loading."""
        return {"training_kind": "observable_maps", "training_schema_version": 1,
                "config": copy.deepcopy(self.config), "data_fingerprint": self.data_fingerprint,
                "device_type": self.device.type, "L": self.L,
                "model_state_dict": _cpu_copy(self.model.state_dict()),
                "optimizer_state_dict": _cpu_copy(self.optimizer.state_dict()),
                "best_state_dict": self.best_state_dict,
                "shuffle_generator_state": self.shuffle_generator.get_state().clone(),
                "permutation": self.permutation.clone(), "cursor": self.cursor,
                "step": self.step, "epoch": self.epoch,
                "hamiltonians_seen": self.hamiltonians_seen,
                "epoch_squared_error": self._epoch_squared_error, "epoch_elements": self._epoch_elements,
                "best_step": self.best_step, "best_epoch": self.best_epoch,
                "best_val_mse": self.best_val_mse, "history": self.history,
                "stopped_early": self.stopped_early,
                "early_stopping_wait": self.early_stopping_wait,
                "early_stopping_reference": self.early_stopping_reference,
                "torch_rng_state": torch.get_rng_state().clone(),
                "cuda_rng_states": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
                if self.device.type == "cuda" else []}

    def _restore(self, state: dict[str, Any]) -> None:
        if state.get("training_kind") != "observable_maps" or type(state.get("training_schema_version")) is not int or state["training_schema_version"] != 1:
            raise ValueError("Unsupported observable-training checkpoint")
        if state.get("data_fingerprint") != self.data_fingerprint or state.get("L") != self.L:
            raise ValueError("Checkpoint training/validation data differ")
        saved_config = _config(state["config"])
        if any(saved_config[key] != self.config[key] for key in self.config if key != "torch_threads"):
            raise ValueError("Checkpoint training settings differ")
        if state.get("device_type") != self.device.type:
            raise ValueError("Exact resume requires the original CPU/CUDA device type")
        expected = self.model.state_dict()
        for name in ("model_state_dict", "best_state_dict"):
            supplied = state[name]
            if set(supplied) != set(expected) or any(
                    not isinstance(supplied[key], torch.Tensor)
                    or supplied[key].dtype != tensor.dtype or supplied[key].shape != tensor.shape
                    or not bool(torch.isfinite(supplied[key]).all().item())
                    for key, tensor in expected.items()):
                raise ValueError(f"Checkpoint {name} has incompatible weights")
        for name in ("step", "epoch", "cursor", "hamiltonians_seen", "epoch_elements", "best_step", "best_epoch", "early_stopping_wait"):
            _integer(state[name], f"checkpoint {name}", 0)
        epoch, step, cursor = state["epoch"], state["step"], state["cursor"]
        count, batch = len(self.train_x), self.config["batch_size_hamiltonians"]
        updates_per_epoch = math.ceil(count / batch)
        permutation = state["permutation"]
        if not isinstance(permutation, torch.Tensor) or permutation.dtype != torch.int64 or permutation.ndim != 1:
            raise ValueError("Checkpoint Hamiltonian permutation is invalid")
        permutation = permutation.cpu().clone()
        if (len(permutation) not in (0, count)
                or (len(permutation) and not torch.equal(torch.sort(permutation).values, torch.arange(count)))
                or cursor > len(permutation) or epoch > self.config["max_epochs"]):
            raise ValueError("Checkpoint Hamiltonian permutation/progress is invalid")
        partial = step - epoch * updates_per_epoch
        expected_cursors = ({0, count} if partial == 0 and step else {min(partial * batch, count)})
        partial_rows = 0 if partial == 0 else cursor
        if (not 0 <= partial < updates_per_epoch or cursor not in expected_cursors
                or (step > 0 and len(permutation) == 0)
                or state["hamiltonians_seen"] != epoch * count + partial_rows
                or state["epoch_elements"] != partial_rows * 2 * self.L * self.L):
            raise ValueError("Checkpoint partial-epoch counters are inconsistent")
        _number(state["epoch_squared_error"], "checkpoint epoch_squared_error", positive=False)
        if partial == 0 and state["epoch_squared_error"] != 0:
            raise ValueError("Checkpoint completed-epoch loss counters are inconsistent")
        best = _number(state["best_val_mse"], "checkpoint best_val_mse", positive=False)
        reference = state["early_stopping_reference"]
        if reference is not None:
            _number(reference, "checkpoint early_stopping_reference", positive=False)
        history = state["history"]
        if not isinstance(history, list) or len(history) != epoch + 1:
            raise ValueError("Checkpoint validation history is inconsistent")
        replay_best = math.inf
        replay_best_epoch = replay_best_step = replay_wait = 0
        replay_reference = None
        replay_stopped = False
        stopping = self.config["early_stopping"]
        for index, event in enumerate(history):
            if (not isinstance(event, dict) or type(event.get("epoch")) is not int
                    or type(event.get("step")) is not int or event["epoch"] != index
                    or event["step"] != index * updates_per_epoch or replay_stopped):
                raise ValueError("Checkpoint validation history has inconsistent steps")
            loss = _number(event.get("validation_mse"), "checkpoint validation_mse", positive=False)
            _number(event.get("train_mse"), "checkpoint train_mse", positive=False)
            if index == 0:
                if event.get("optimizer_epoch_mse") is not None:
                    raise ValueError("Checkpoint initial check has an optimizer loss")
            else:
                _number(event.get("optimizer_epoch_mse"), "checkpoint optimizer_epoch_mse", positive=False)
            improved = loss < replay_best
            if improved:
                replay_best, replay_best_epoch = loss, index
                replay_best_step = index * updates_per_epoch
            if index <= stopping["min_epochs"]:
                replay_wait, replay_reference = 0, replay_best
            elif stopping["enabled"]:
                if replay_reference is None or loss < replay_reference - stopping["min_delta"]:
                    replay_wait, replay_reference = 0, loss
                else:
                    replay_wait += 1
                    replay_stopped = replay_wait >= stopping["patience_epochs"]
            reason = ("early_stopping" if replay_stopped else
                      "max_epochs" if index >= self.config["max_epochs"] else None)
            expected_event = {"hamiltonians_seen": index * count,
                              "best_validation_mse": replay_best, "best_step": replay_best_step,
                              "best_epoch": replay_best_epoch, "improved": improved,
                              "early_stopping_wait": replay_wait,
                              "early_stopping_reference": replay_reference,
                              "stopped_early": replay_stopped, "stop_reason": reason}
            if any(event.get(key) != value for key, value in expected_event.items()):
                raise ValueError("Checkpoint validation history fails selection/stopping replay")
        if (best != replay_best or state["best_epoch"] != replay_best_epoch
                or state["best_step"] != replay_best_step):
            raise ValueError("Checkpoint best-model selection is inconsistent")
        if (not isinstance(state["stopped_early"], bool)
                or state["stopped_early"] != replay_stopped
                or state["early_stopping_wait"] != replay_wait
                or reference != replay_reference or (replay_stopped and partial != 0)):
            raise ValueError("Checkpoint early-stopping state is inconsistent")
        self.model.load_state_dict(state["model_state_dict"])
        supplied_optimizer = state["optimizer_state_dict"]
        optimizer_groups = supplied_optimizer.get("param_groups", [])
        expected_groups = self.optimizer.state_dict()["param_groups"]
        if (len(optimizer_groups) != len(expected_groups)
                or any(set(group) != set(expected_group)
                       or any(group[key] != expected_group[key] for key in group)
                       for group, expected_group in zip(optimizer_groups, expected_groups))):
            raise ValueError("Checkpoint Adam hyperparameters differ")
        parameter_ids = [index for group in optimizer_groups for index in group["params"]]
        optimizer_state = supplied_optimizer.get("state", {})
        if set(optimizer_state) != (set(parameter_ids) if step else set()):
            raise ValueError("Checkpoint Adam state does not match progress")
        for index, parameter in zip(parameter_ids, self.model.parameters()):
            if step == 0:
                break
            adam = optimizer_state[index]
            if set(adam) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("Checkpoint Adam moment keys are incompatible")
            for name in ("exp_avg", "exp_avg_sq"):
                if (name not in adam or not isinstance(adam[name], torch.Tensor)
                        or adam[name].dtype != parameter.dtype
                        or adam[name].shape != parameter.shape or not bool(torch.isfinite(adam[name]).all().item())):
                    raise ValueError("Checkpoint Adam moments are incompatible")
            if bool(torch.any(adam["exp_avg_sq"] < 0).item()):
                raise ValueError("Checkpoint Adam squared moments must be nonnegative")
            adam_step = adam.get("step")
            if (not isinstance(adam_step, torch.Tensor) or adam_step.ndim != 0
                    or float(adam_step.item()) != step):
                raise ValueError("Checkpoint Adam update counters are incompatible")
        self.optimizer.load_state_dict(_cpu_copy(supplied_optimizer))
        self._best_state_dict = _cpu_copy(state["best_state_dict"])
        self.shuffle_generator.set_state(state["shuffle_generator_state"].cpu())
        self.permutation, self._device_permutation = permutation, permutation.to(self.device)
        self.cursor, self.step, self.epoch = cursor, step, epoch
        self.hamiltonians_seen = state["hamiltonians_seen"]
        self._epoch_squared_error, self._epoch_elements = state["epoch_squared_error"], state["epoch_elements"]
        self.best_val_mse, self.best_step, self.best_epoch = best, state["best_step"], state["best_epoch"]
        self._history = copy.deepcopy(history)
        self.stopped_early = state["stopped_early"]
        self.early_stopping_wait, self.early_stopping_reference = state["early_stopping_wait"], reference
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_states"]])
