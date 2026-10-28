#!/usr/bin/env python3
"""Prepare portable ignored inputs; preserve original banks, models and notes."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from quantum_lattice.models.final_campaign import atomic_json, artifact, read, sha, validate_config


def copy_verified(source, target):
    expected = sha(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != expected:
            raise ValueError(f"Refusing to overwrite different staged bytes: {target}")
    else:
        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        if sha(temporary) != expected:
            raise ValueError("Staging copy verification failed")
        temporary.replace(target)
    return expected


def stage(config, *, bundle=None, gpu_uuid=None):
    config = validate_config(config)
    prior = artifact(ROOT, config["prior_manifest_root"])
    original = ROOT/"codex material/artifacts/phase1_sample_generation/campaigns/phase1_paper_domain"
    for L in config["sizes"]:
        copy_verified(original/f"L{L}"/"run"/"manifest.json", prior/f"L{L}.json")
    if gpu_uuid is not None:
        worker = json.loads(json.dumps(config))
        worker["hardware"]["gpu_uuid"] = gpu_uuid
        worker = validate_config(worker)
        path = artifact(ROOT, "codex material/artifacts/final_benchmark/inputs/worker-config.json")
        if path.exists() and read(path) != worker:
            raise ValueError("Refusing to change an existing private worker configuration")
        atomic_json(path, worker)
    raw = config["raw_panel"]
    if raw["enabled"]:
        destination = artifact(ROOT, raw["source_root"])
        source = ROOT/"codex material/artifacts/snapshot_learning/scaling-review-20261027/retrieved/output/models"
        origin = source.parents[1]/"export-manifest.json"
        exported = read(origin)
        pins, tasks = {}, []
        for seed in raw["seeds"]:
            for method in raw["methods"]:
                key = f"L4-H7452-M256-seed{seed}-{method}"
                done = read(source/key/"done.json")
                if exported["files"][f"output/models/{key}/done.json"]["sha256"] != sha(source/key/"done.json"):
                    raise ValueError("Raw completion marker differs from the verified clean export")
                if done["task"]["key"] != key or done["hashes"]["best.pt"] != sha(source/key/"best.pt"):
                    raise ValueError("Original raw selected-model binding differs")
                tasks.append(done["task"])
                for name in ("best.pt", "done.json", "evaluation.json"):
                    if name != "done.json" and done["hashes"].get(name) != sha(source/key/name):
                        raise ValueError(f"Original committed raw output changed: {key}/{name}")
                    if name == "evaluation.json" and read(source/key/name).get("task") != done["task"]:
                        raise ValueError("Original raw evaluation identity differs")
                    pins[f"{key}/{name}"] = copy_verified(source/key/name, destination/key/name)
        pins["origin-export-manifest.json"] = copy_verified(origin, destination/"origin-export-manifest.json")
        manifest = {"schema_version": 1, "scope": "immutable selected clean L4 raw controls; no optimizer resume or old validation data",
                    "tasks": tasks, "files": pins}
        if (destination/"manifest.json").exists() and read(destination/"manifest.json") != manifest:
            raise ValueError("Staged raw reference manifest changed")
        atomic_json(destination/"manifest.json", manifest)
    if bundle is None:
        return
    bundle = artifact(ROOT, bundle)
    if bundle.exists():
        raise FileExistsError("Use a new bundle path, preserving earlier transfer evidence")
    files = list((ROOT/"src/quantum_lattice").rglob("*.py"))
    files += [p for p in (ROOT/"experiments/final_benchmark").iterdir()
              if p.is_file() and p.suffix in {".py", ".md", ".sh"}]
    files += [ROOT/"configs/final_benchmark.json", ROOT/"pyproject.toml", ROOT/".gitignore"]
    for subtree in (config["feature_cache"], config["prior_manifest_root"], raw.get("source_root", "")):
        if subtree:
            files += [p for p in artifact(ROOT, subtree).rglob("*") if p.is_file() and not p.name.startswith(".")]
    private = artifact(ROOT, "codex material/artifacts/final_benchmark/inputs/worker-config.json")
    if private.exists():
        files.append(private)
    files = sorted(set(files))
    manifest = {"schema_version": 1, "scope": "portable source snapshot plus compact non-test features, prior UID metadata and raw best-only controls",
                "files": {str(p.relative_to(ROOT)): sha(p) for p in files},
                "excludes": ["old test payloads", "original raw training records", "private literature/paper notes", "user MLP config", "credentials", "full-run outputs"]}
    manifest_path = bundle.with_suffix(".manifest.json")
    atomic_json(manifest_path, manifest)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "w:gz") as archive:
        for p in files:
            archive.add(p, arcname=p.relative_to(ROOT), recursive=False)
        archive.add(manifest_path, arcname="transfer-manifest.json", recursive=False)
    atomic_json(bundle.with_suffix(".sha256.json"), {"sha256": sha(bundle), "bytes": bundle.stat().st_size,
                "files": len(files), "uncompressed_bytes": sum(p.stat().st_size for p in files)})
    print(f"Staged {len(files)} files; bundle {bundle.stat().st_size/2**20:.1f} MiB", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/final_benchmark.json")
    parser.add_argument("--bundle", help="Fresh path inside ignored final_benchmark artifacts")
    parser.add_argument("--gpu-uuid", help="Verified GPU binding stored only in the ignored worker configuration")
    args = parser.parse_args()
    stage(read(ROOT/args.config), bundle=args.bundle, gpu_uuid=args.gpu_uuid)
