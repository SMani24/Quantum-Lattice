#!/usr/bin/env python3
"""CPU-only checks and non-test cache inspection for observable-map models."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from quantum_lattice.data.estimator_features import load_feature_views
from quantum_lattice.data.observable_maps import (
    INPUT_CHANNELS, TARGET_CHANNELS, ObservableMaps, edge_rows_to_maps,
    flatten_edge_maps, map_learning_data, maps_to_edge_rows,
)
from quantum_lattice.models.mlp import PaperMLP
from quantum_lattice.models.observable_cnn import ObservableMapMLP, PeriodicObservableCNN

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "codex material/artifacts/observable_cnn"
DEFAULT_CONFIG = ROOT / "configs/observable_cnn.json"


def _integer(value, minimum, maximum, label):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [{minimum},{maximum}]")
    return value


def validate_config(config):
    if (not isinstance(config, dict) or type(config.get("schema_version")) is not int
            or config["schema_version"] != 1):
        raise ValueError("unsupported foundation configuration")
    if set(config) != {"schema_version","scope","feature_cache","architecture","checks"}:
        raise ValueError("declare only the foundation configuration fields")
    if not isinstance(config["scope"], str) or not config["scope"].strip():
        raise ValueError("scope must be a nonempty description of the foundation check")
    architecture, checks = config["architecture"], config["checks"]
    if not isinstance(architecture,dict) or set(architecture)!={"hidden_channels","kernel_size"}:
        raise ValueError("declare only hidden_channels and kernel_size for the architecture")
    if not isinstance(checks,dict) or set(checks)!={"sizes","seed","synthetic_hamiltonians","optimizer_steps","learning_rate"}:
        raise ValueError("declare the bounded synthetic-check settings")
    widths = architecture["hidden_channels"]
    if not isinstance(widths, list) or not 1 <= len(widths) <= 5:
        raise ValueError("declare one to five hidden channel widths")
    for width in widths:
        _integer(width, 1, 128, "hidden width")
    if architecture["kernel_size"] not in (1, 3) or type(architecture["kernel_size"]) is not int:
        raise ValueError("kernel_size must be 1 or 3")
    if not isinstance(checks["sizes"], list) or not checks["sizes"] or len(set(checks["sizes"])) != len(checks["sizes"]):
        raise ValueError("sizes must be a nonempty distinct list")
    for size in checks["sizes"]:
        _integer(size, 3, 32, "check size")
    _integer(checks["seed"], 0, 2**63-1, "seed")
    _integer(checks["synthetic_hamiltonians"], 1, 8, "synthetic_hamiltonians")
    _integer(checks["optimizer_steps"], 1, 50, "optimizer_steps")
    rate = checks["learning_rate"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not np.isfinite(rate) or not 0 < rate <= .01:
        raise ValueError("synthetic learning_rate must be in (0,.01]")
    if not isinstance(config.get("feature_cache"), str) or not config["feature_cache"]:
        raise ValueError("feature_cache is required")
    return config


def _tensor(values):
    # The adapter intentionally owns read-only arrays; copy into writable CPU storage.
    return torch.from_numpy(np.array(values, dtype=np.float32, copy=True)).to("cpu")


def _assert_roundtrip(rows, maps):
    recovered = maps_to_edge_rows(maps)
    for key in ("x", "y", "hamiltonian_ids", "hamiltonian_uids", "edge_ids"):
        if not np.array_equal(getattr(rows, key), getattr(recovered, key)):
            raise ValueError(f"lossless roundtrip failed for {key}")


def synthetic_maps(L, count, seed):
    """Artificial observables and a local regression target; no physics claim."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(.1, 1.2, (count, 3, L, L)).astype(np.float32)
    stars = x[:, 0]
    y = np.stack((.2+.25*stars+.3*np.roll(stars,-1,axis=2)+.1*x[:,1],
                  .2+.25*stars+.3*np.roll(stars,-1,axis=1)+.1*x[:,2]), axis=1)
    return ObservableMaps(x, y.astype(np.float32), np.array([f"synthetic-{n}" for n in range(count)]),
                          np.array([f"L{L}-seed{seed}-H{n}" for n in range(count)]),
                          np.arange(count*2*L*L).reshape(count,-1), L)


def check(config):
    """Bounded software checks only; all tensors/optimizer state remain on CPU."""
    settings, architecture = config["checks"], config["architecture"]
    torch.set_num_threads(1)
    cases = []
    for L in settings["sizes"]:
        maps = synthetic_maps(L, settings["synthetic_hamiltonians"], settings["seed"]+L)
        rows = maps_to_edge_rows(maps)
        _assert_roundtrip(rows, edge_rows_to_maps(rows, L))
        inputs, targets = _tensor(maps.x), _tensor(maps.y)
        torch.random.default_generator.manual_seed(settings["seed"]+L)
        local = PaperMLP().to("cpu")
        wrapper = ObservableMapMLP(local)
        with torch.no_grad():
            original = local(_tensor(rows.x)).numpy()
            mapped = flatten_edge_maps(maps, wrapper(inputs).numpy())
        error = float(np.max(np.abs(original-mapped)))
        if error > 1e-6:
            raise ValueError("map MLP differs from the original row MLP")
        cnn = PeriodicObservableCNN(**architecture).to("cpu")
        with torch.no_grad():
            predicted = cnn(inputs)
            shifted = cnn(torch.roll(inputs, (1,-1), (-2,-1)))
            periodic_error = float(torch.max(torch.abs(shifted-torch.roll(predicted,(1,-1),(-2,-1)))))
        if periodic_error > 1e-5:
            raise ValueError("periodic translation check failed")
        losses = {}
        for name, model in (("cnn", cnn), ("local_mlp", wrapper)):
            optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
            history = []
            for _ in range(settings["optimizer_steps"]):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.mean((model(inputs)-targets).square())
                loss.backward()
                if not torch.isfinite(loss) or any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                                                   for parameter in model.parameters()):
                    raise ValueError("synthetic forward/backward is nonfinite")
                history.append(float(loss.detach()))
                optimizer.step()
            with torch.no_grad():
                final = float(torch.mean((model(inputs)-targets).square()))
            if not np.isfinite(final):
                raise ValueError("nonfinite final synthetic loss")
            losses[name] = {"initial_loss":history[0],"final_loss":final,"optimizer_steps":len(history),
                            "loss_history":history,"parameter_count":sum(p.numel() for p in model.parameters())}
        cases.append({"L":L,"input_shape":list(inputs.shape),"target_shape":list(targets.shape),
                      "roundtrip_exact":True,"original_vs_map_mlp_max_abs":error,
                      "translation_max_abs":periodic_error,"synthetic_only":True,"models":losses})
    return {"schema_version":1,"scope":"CPU-only software checks on artificial data; no research results",
            "cases":cases,"device":"cpu","research_training_performed":False,"real_bank_loaded":False,
            "input_channels":list(INPUT_CHANNELS),"target_channels":list(TARGET_CHANNELS)}


def inspect(config, L, feature_dir=None):
    """Verify lossless maps for every non-test cached view; no optimizer steps."""
    directory = Path(feature_dir or config["feature_cache"])
    if not directory.is_absolute():
        directory = ROOT / directory
    views = load_feature_views(directory, L)
    cases = []
    for view, data in views.items():
        mapped = map_learning_data(data)
        for split in ("train", "validation"):
            rows, maps = getattr(data, split), getattr(mapped, split)
            _assert_roundtrip(rows, maps)
            cases.append({"view":view,"split":split,"hamiltonians":len(maps),"edge_rows":len(rows),
                          "input_shape":list(maps.x.shape),"target_shape":list(maps.y.shape),
                          "roundtrip_exact":True,"source_information_unchanged":True,
                          "training_eligible":mapped.provenance.get("training_eligible",view!="exact"),
                          "maximum_input":float(maps.x.max()),"minimum_input":float(maps.x.min())})
    manifest=directory/"manifest.json"
    return {"schema_version":1,"scope":"non-test feature layout/identity check, not model evaluation",
            "L":L,"cases":cases,"feature_root":str(directory),
            "feature_manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "test_payload_loaded":False,"spin_records_loaded":False,"optimizer_steps":0,
            "device":"cpu","input_channels":list(INPUT_CHANNELS),"target_channels":list(TARGET_CHANNELS)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("check","inspect"), default="check")
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    parser.add_argument("--L",type=int,default=4,help="cache size for inspect (default4)")
    parser.add_argument("--feature-dir",type=Path,help="existing paired feature cache for inspect")
    parser.add_argument("--output",type=Path,help="new directory below the ignored observable_cnn artifact root")
    args=parser.parse_args(argv)
    config=validate_config(json.loads(args.config.read_text()))
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output=(args.output or ARTIFACT_ROOT/f"{args.command}-{stamp}").resolve()
    if output==ARTIFACT_ROOT.resolve() or not output.is_relative_to(ARTIFACT_ROOT.resolve()):
        raise ValueError("output must be a new subdirectory of the observable_cnn artifact root")
    if output.exists():
        raise ValueError("output already exists; choose a new directory")
    started=time.perf_counter()
    result=check(config) if args.command=="check" else inspect(config,args.L,args.feature_dir)
    result["elapsed_seconds"]=time.perf_counter()-started
    output.mkdir(parents=True)
    paths=tuple(ROOT/"src/quantum_lattice"/name for name in
                ("data/observable_maps.py","models/observable_cnn.py","models/mlp.py",
                 "data/estimator_features.py","data/mlp_data.py","lattice/builders.py",
                 "lattice/topology.py","lattice/square.py"))+(Path(__file__),)
    provenance={"source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                "config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "torch":str(torch.__version__),"numpy":np.__version__,"python":sys.version,
                "device":"cpu","gpu_work":False,"research_training_performed":False}
    for name,value in (("checks.json",result),("config.json",config),("provenance.json",provenance)):
        (output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    lines=["# Observable-map CNN foundation check","",result["scope"],"",
           f"Device: CPU. Elapsed: {result['elapsed_seconds']:.3f}s. No research training or GPU work.","",
           "Inputs: A, neighboring AA-horizontal, neighboring AA-vertical. Targets: horizontal/vertical |b_z|.","",
           "Details: `checks.json`; effective settings: `config.json`; sources/environment: `provenance.json`.","",
           "These are implementation checks, not quantum-learning accuracy or a completed architecture comparison."]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(result["scope"])
    for case in result["cases"]:
        if "view" in case:
            print(f"  {case['view']:5} {case['split']:10}: {case['hamiltonians']} maps; "
                  f"inputs {case['input_shape']}; exact roundtrip")
        else:
            print(f"  L={case['L']}: original/map MLP max delta {case['original_vs_map_mlp_max_abs']:.3g}; "
                  f"periodic delta {case['translation_max_abs']:.3g}; synthetic gradient checks pass")
    print(f"Results: {output/'RESULTS.md'}")
    return 0


if __name__=="__main__":
    try:
        raise SystemExit(main())
    except (ValueError,TypeError,KeyError,OSError) as error:
        print(f"Observable CNN check stopped: {error}",file=sys.stderr)
        raise SystemExit(1)
