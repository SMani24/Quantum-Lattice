"""Read-only D5 scoring of two frozen, matched estimator MLP campaigns.

The supplied signs are scoring assistance to magnitude-only predictions, never
learned sign recovery. All original crossed validation views and seeds remain
visible. This module does no training, inference, sampling or test-bank access.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from quantum_lattice.data.hamiltonian_scores import (
    load_prediction_maps, load_validation_fields, reference_sign_lift,
)
from quantum_lattice.lattice import square_lattice
from quantum_lattice.metrics.hamiltonian_batch import CompiledHamiltonianMetric

ARTIFACT_ROOT = Path("codex material/artifacts/hamiltonian_scores")
METRICS = ("two_channel_full", "two_channel_traceless", "z_branch_full", "z_branch_traceless")
VIEWS = ("f", "g", "exact")
CONTEXTS = (*VIEWS, "matched_pipeline")


def _json(path):
    return json.loads(Path(path).read_text())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _inside(root, name):
    root = Path(root).resolve()
    result = (root / name).resolve()
    if not result.is_relative_to(root):
        raise ValueError("input path escapes the repository/artifact root")
    return result


def _snapshot(directory):
    directory = Path(directory).resolve()
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            if not path.resolve().is_relative_to(directory):
                raise ValueError("campaign contains a file link outside its root")
            result[str(path.relative_to(directory))] = _sha(path)
    return result


def validate_config(config):
    required = {"schema_version", "campaigns", "split", "sign_source", "prediction_policy",
                "primary_metric", "include_zero_field_baseline", "batch_size"}
    if (not isinstance(config, dict) or set(config) != required
            or type(config["schema_version"]) is not int or config["schema_version"] != 1):
        raise ValueError("unsupported Hamiltonian-score configuration")
    if (config["split"] != "validation" or config["sign_source"] != "reference_sign_oracle"
            or config["prediction_policy"] != "raw_unclipped"
            or config["primary_metric"] != METRICS[0]
            or config["include_zero_field_baseline"] is not True):
        raise ValueError("declare validation, raw oracle predictions, primary D5 and zero baseline")
    if type(config["batch_size"]) is not int or not 1 <= config["batch_size"] <= 1024:
        raise ValueError("batch_size must be an integer in [1,1024]")
    campaigns = config["campaigns"]
    if not isinstance(campaigns, list) or len(campaigns) != 2:
        raise ValueError("declare the original and continued matched campaigns")
    for campaign in campaigns:
        if (not isinstance(campaign, dict) or set(campaign) != {"label", "run_dir"}
                or not isinstance(campaign["label"], str)
                or re.fullmatch(r"[a-z][a-z0-9_]{0,31}", campaign["label"]) is None
                or not isinstance(campaign["run_dir"], str) or not campaign["run_dir"]):
            raise ValueError("campaign labels/paths are invalid")
    if len({c["label"] for c in campaigns}) != 2 or len({c["run_dir"] for c in campaigns}) != 2:
        raise ValueError("campaigns must have distinct labels and directories")
    return config


def _campaign(root, entry):
    directory = _inside(root, entry["run_dir"])
    if not directory.is_relative_to((root / "codex material/artifacts").resolve()):
        raise ValueError("campaign must be an existing ignored artifact directory")
    plan = _json(directory / "plan.json")
    digest = _digest(plan)
    if (_json(directory / "plan-sha256.json") != {"sha256": digest}
            or type(plan.get("schema_version")) is not int or plan["schema_version"] != 1):
        raise ValueError("frozen campaign plan hash/schema mismatch")
    for name, expected in plan["source_hashes"].items():
        if _sha(_inside(root, name)) != expected:
            raise ValueError(f"frozen campaign source changed: {name}")
    if "parent" in plan:
        parent = _inside(root, plan["parent"]["run"])
        original = _json(parent / "plan.json")
        if _digest(original) != plan["parent"]["plan_sha256"] or plan["tasks"] != original["tasks"]:
            raise ValueError("continuation parent/task identity differs")
        for name, expected in plan["parent"]["files"].items():
            if _sha(_inside(parent, name)) != expected:
                raise ValueError(f"frozen parent file changed: {name}")
        features = parent / "features"
        expected_feature_hash = plan["parent"]["files"]["features/manifest.json"]
    else:
        features = directory / "features"
        expected_feature_hash = plan["feature_manifest_sha256"]
    if _sha(features / "manifest.json") != expected_feature_hash:
        raise ValueError("campaign feature manifest binding differs")
    sizes, seeds = plan["config"]["sizes"], plan["config"]["seeds"]
    if (not sizes or len(set(sizes)) != len(sizes)
            or any(type(L) is not int or L < 3 for L in sizes)
            or not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("invalid frozen sizes/seeds")
    expected_tasks = [{"L": L, "seed": seed, "view": view,
                       "key": f"L{L}-seed{seed}-{view}"}
                      for L in sizes for seed in seeds for view in ("f", "g")]
    if plan["tasks"] != expected_tasks or plan["config"]["mode"] != "all_edges":
        raise ValueError("expected every predeclared all-edge f/g task")
    return {"label": entry["label"], "directory": directory, "plan": plan,
            "plan_digest": digest, "feature_dir": features,
            "feature_manifest_sha256": expected_feature_hash, "snapshot": _snapshot(directory)}


def score_fields(reference_fields, raw_predictions, compiled, batch_size):
    """Raw per-H scores, never a distance between averaged field vectors."""
    recovered = reference_sign_lift(reference_fields, raw_predictions)
    result = {name: np.empty(len(recovered), dtype=np.float64) for name in METRICS}
    for start in range(0, len(recovered), batch_size):
        end = min(start + batch_size, len(recovered))
        for model, metric in compiled.items():
            scores = metric.distances(reference_fields[start:end], recovered[start:end])
            for identity, values in scores.items():
                result[f"{model}_{identity}"][start:end] = values
    if any(not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 2)
           for values in result.values()):
        raise ValueError("nonfinite or out-of-range D5 scores")
    invariance = float(np.max(np.abs(result[METRICS[1]] - result[METRICS[3]])))
    if invariance > 1e-12:
        raise ValueError("zero-X traceless model invariance failed")
    difference = recovered - reference_fields
    return result, {"mse": np.mean(difference * difference, axis=1),
                    "mae": np.mean(np.abs(difference), axis=1)}, invariance


def _csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_scores(campaigns, sizes, seeds, scores):
    """Equal Hamiltonian means, then equal paired seeds; descriptive only."""
    groups, pairs, contrasts, parent_changes = [], [], [], []
    for campaign in campaigns:
        for L in sizes:
            for training in ("f", "g"):
                for view in VIEWS:
                    for metric in METRICS:
                        values = np.array([np.mean(scores[campaign,L,seed,training,view][metric])
                                           for seed in seeds])
                        groups.append({"campaign": campaign, "L": L, "training_view": training,
                                       "input_view": view, "metric": metric,
                                       "mean_d5": float(values.mean()),
                                       "seed_minimum": float(values.min()),
                                       "seed_maximum": float(values.max()), "seeds": len(seeds)})
            for context in CONTEXTS:
                for metric in METRICS:
                    first = np.stack([scores[campaign,L,seed,"f","f" if context == "matched_pipeline" else context][metric]
                                      for seed in seeds])
                    second = np.stack([scores[campaign,L,seed,"g","g" if context == "matched_pipeline" else context][metric]
                                       for seed in seeds])
                    a, b = first.mean(axis=1), second.mean(axis=1)
                    for index, seed in enumerate(seeds):
                        pairs.append({"campaign": campaign,"L": L,"seed": seed,"context": context,
                                      "metric": metric,"f_mean_d5": float(a[index]),
                                      "g_mean_d5": float(b[index]),"g_minus_f": float(b[index]-a[index]),
                                      "relative_reduction": float(1-b[index]/a[index]) if a[index] else None})
                    contrasts.append({"campaign": campaign,"L": L,"context": context,"metric": metric,
                                      "f_mean_d5": float(a.mean()),"g_mean_d5": float(b.mean()),
                                      "g_minus_f": float(b.mean()-a.mean()),
                                      "relative_reduction": float(1-b.mean()/a.mean()) if a.mean() else None,
                                      "seeds_g_better": int(np.sum(b<a)),"paired_seeds": len(seeds),
                                      "hamiltonians_g_better_after_seed_average": int(np.sum(second.mean(axis=0)<first.mean(axis=0))),
                                      "hamiltonians": first.shape[1]})
    for L in sizes:
        for seed in seeds:
            for training in ("f", "g"):
                for view in VIEWS:
                    for metric in METRICS:
                        a = float(scores[campaigns[0],L,seed,training,view][metric].mean())
                        b = float(scores[campaigns[1],L,seed,training,view][metric].mean())
                        parent_changes.append({"L": L,"seed": seed,"training_view": training,
                                               "input_view": view,"metric": metric,
                                               "parent_mean_d5": a,"continued_mean_d5": b,
                                               "difference": b-a,"relative_reduction": 1-b/a if a else None})
    return groups, pairs, contrasts, parent_changes


def evaluate_campaigns(repo_root, config, *, output=None, output_callback=print):
    """Score all frozen outputs in a new ignored run without changing inputs."""
    root = Path(repo_root).resolve()
    validate_config(config)
    artifact_root = (root / ARTIFACT_ROOT).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(output).resolve() if output is not None else artifact_root / f"evaluation-{stamp}"
    if directory == artifact_root or not directory.is_relative_to(artifact_root):
        raise ValueError("output must be a new subdirectory of the ignored Hamiltonian-score root")
    if directory.exists():
        raise ValueError("output already exists; choose a new run directory")
    started = time.perf_counter()
    campaigns = [_campaign(root, entry) for entry in config["campaigns"]]
    if (campaigns[0]["directory"] == campaigns[1]["directory"]
            or campaigns[0]["plan"]["tasks"] != campaigns[1]["plan"]["tasks"]
            or campaigns[0]["feature_dir"] != campaigns[1]["feature_dir"]):
        raise ValueError("campaigns must be distinct, matching tasks and the same bound feature cache")
    origin = campaigns[1]["plan"].get("parent")
    if ("parent" in campaigns[0]["plan"] or not isinstance(origin, dict)
            or _inside(root, origin["run"]) != campaigns[0]["directory"]
            or origin["plan_sha256"] != campaigns[0]["plan_digest"]):
        raise ValueError("the configured continuation must descend from the configured original campaign")
    sizes, seeds = campaigns[0]["plan"]["config"]["sizes"], campaigns[0]["plan"]["config"]["seeds"]
    references = {L: load_validation_fields(campaigns[0]["feature_dir"], L) for L in sizes}
    if any(ref.source_manifest_sha256 != campaigns[0]["feature_manifest_sha256"] for ref in references.values()):
        raise ValueError("signed reference cache differs from frozen model data")
    compiled = {L: {model: CompiledHamiltonianMetric(square_lattice(L), model=model)
                    for model in ("two_channel", "z_branch")} for L in sizes}
    source_paths = set(campaigns[0]["plan"]["source_hashes"]) | set(campaigns[1]["plan"]["source_hashes"])
    source_paths |= {"src/quantum_lattice/data/hamiltonian_scores.py",
                     "src/quantum_lattice/metrics/hamiltonian.py",
                     "src/quantum_lattice/metrics/hamiltonian_batch.py",
                     "src/quantum_lattice/metrics/hamiltonian_evaluation.py",
                     "experiments/hamiltonian_scores/run.py"}
    sources = {name: _sha(_inside(root, name)) for name in sorted(source_paths)}
    plan = {"schema_version":1,"config":config,"metrics":list(METRICS),
            "primary_metric":METRICS[0],"scope":"development validation; reference-sign oracle",
            "true_field_precision":"original seeded float64; float32 abs labels verified",
            "recovered_field_role":"estimate of original Z fields, not residual after correction",
            "x_fields":"known zero in true and reconstructed Hamiltonians",
            "sign_lift":"sign(true_b_z) * raw_prediction; exact zero references map to zero",
            "figure_rescaling":"none; raw unsquared D5 in [0,2]",
            "aggregation":"arithmetic Hamiltonian mean then equal arithmetic seed mean",
            "zero_field_baseline":"estimated Z fields zero, same X/model/identity convention",
            "campaigns":[{"label":c["label"],"run_dir":str(c["directory"].relative_to(root)),
                           "canonical_plan_sha256":c["plan_digest"],"files":c["snapshot"]}
                          for c in campaigns], "source_sha256":sources,
            "feature_manifest_sha256":campaigns[0]["feature_manifest_sha256"],
            "numpy":np.__version__,"compiled_basis_counts":{
                str(L):{name:metric.n_coefficients for name,metric in compiled[L].items()} for L in sizes}}
    # Freeze all conventions, input bytes and sources before the first D5 call.
    directory.mkdir(parents=True)
    _write(directory/"plan.json",plan)
    _write(directory/"config.json",config)
    _write(directory/"plan-sha256.json",{"sha256":_digest(plan)})
    score_cache, model_rows, baseline_rows = {}, [], []
    maximum_invariance = 0.0
    arrays = {}
    for L, ref in references.items():
        arrays.update({f"L{L}__true_b_z":ref.true_b_z,f"L{L}__ids":ref.hamiltonian_ids,
                       f"L{L}__uids":ref.hamiltonian_uids})
        zero, _, residual = score_fields(ref.true_b_z,np.zeros_like(ref.true_b_z),compiled[L],config["batch_size"])
        maximum_invariance = max(maximum_invariance,residual)
        for index,name in enumerate(ref.hamiltonian_ids):
            baseline_rows.append({"L":L,"hamiltonian_id":name,"hamiltonian_uid":ref.hamiltonian_uids[index],
                                  **{metric:float(zero[metric][index]) for metric in METRICS}})
    np.savez_compressed(directory/"reference-fields.npz",**arrays)
    _csv(directory/"zero-field-baseline.csv",baseline_rows)
    fields = ("campaign","key","L","seed","training_view","input_view","hamiltonian_id",
              "hamiltonian_uid",*METRICS,"field_mse","field_mae")
    with (directory/"hamiltonian-scores.csv").open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for campaign in campaigns:
            for task in campaign["plan"]["tasks"]:
                ref = references[task["L"]]
                predictions = load_prediction_maps(campaign["directory"]/"models"/task["key"],task,ref,
                                                   plan_sha256=campaign["plan_digest"])
                for view, values in predictions.items():
                    scores, errors, residual = score_fields(ref.true_b_z,values,compiled[task["L"]],config["batch_size"])
                    maximum_invariance = max(maximum_invariance,residual)
                    key = (campaign["label"],task["L"],task["seed"],task["view"],view)
                    score_cache[key] = scores
                    record = {"campaign":campaign["label"],"key":task["key"],"L":task["L"],
                              "seed":task["seed"],"training_view":task["view"],"input_view":view}
                    for index,name in enumerate(ref.hamiltonian_ids):
                        writer.writerow({**record,"hamiltonian_id":name,"hamiltonian_uid":ref.hamiltonian_uids[index],
                                         **{metric:float(scores[metric][index]) for metric in METRICS},
                                         "field_mse":float(errors["mse"][index]),"field_mae":float(errors["mae"][index])})
                    model_rows.append({**record,"hamiltonians":len(ref.true_b_z),
                                       **{metric:float(scores[metric].mean()) for metric in METRICS},
                                       "field_mse":float(errors["mse"].mean()),
                                       "field_rmse":float(np.sqrt(errors["mse"].mean())),
                                       "field_mae":float(errors["mae"].mean()),
                                       "negative_raw_predictions":int(np.sum(values<0))})
                output_callback(f"Scored {campaign['label']}/{task['key']}: all three validation input views")
    groups,pairs,contrasts,parent_changes = summarize_scores([c["label"] for c in campaigns],sizes,seeds,score_cache)
    for name,rows in (("model-scores.csv",model_rows),("group-means.csv",groups),
                      ("paired-seeds.csv",pairs),("contrasts.csv",contrasts),
                      ("parent-comparison.csv",parent_changes)):
        _csv(directory/name,rows)
    for campaign in campaigns:
        if _snapshot(campaign["directory"]) != campaign["snapshot"]:
            raise ValueError("research inputs changed during scoring; no completion record written")
    if any(_sha(_inside(root,name))!=digest for name,digest in sources.items()):
        raise ValueError("evaluation source changed during scoring")
    summary = {"schema_version":1,"state":"complete","scope":plan["scope"],
               "primary_metric":METRICS[0],"sign_source":config["sign_source"],
               "model_input_cells":len(model_rows),"scored_hamiltonian_rows":sum(r["hamiltonians"] for r in model_rows),
               "sizes":sizes,"seeds":seeds,"contrasts":contrasts,
               "maximum_zero_x_traceless_invariance_error":maximum_invariance,
               "zero_field_baseline_mean":{str(L):{metric:float(np.mean([r[metric] for r in baseline_rows if r["L"]==L]))
                                                     for metric in METRICS} for L in sizes},
               "true_zero_reference_edges":sum(int(np.sum(r.true_b_z==0)) for r in references.values()),
               "input_files_unchanged":True,"training_or_new_sampling":False,"gpu_work":False,
               "test_payload_access":False,"elapsed_seconds":time.perf_counter()-started}
    _write(directory/"summary.json",summary)
    lines=["# D5 scores of frozen MLP predictions","",f"State: complete. {len(model_rows)} model/input cells.","",
           "Primary: raw unsquared two_channel/full D5; known reference-sign oracle. Validation only.","",
           "Each Hamiltonian is scored before averaging Hamiltonians and paired seeds. No clipping or rescaling.","",
           "| Campaign | L | f→f mean D5 | g→g mean D5 | Relative reduction |","|---|---:|---:|---:|---:|"]
    for row in contrasts:
        if row["context"]=="matched_pipeline" and row["metric"]==METRICS[0]:
            percent=f"{100*row['relative_reduction']:.4f}%" if row["relative_reduction"] is not None else "undefined"
            lines.append(f"| {row['campaign']} | {row['L']} | {row['f_mean_d5']:.8f} | {row['g_mean_d5']:.8f} | {percent} |")
    lines += ["","The four convention columns are sensitivity checks; the two traceless scores coincide at zero X fields.","",
              "Files: plan/config/provenance bindings; reference-fields.npz; hamiltonian-scores.csv; model-scores.csv;",
              "group-means.csv; paired-seeds.csv; contrasts.csv; parent-comparison.csv; zero-field-baseline.csv; summary.json.","",
              "Raw outputs, sources and feature caches remain unchanged. No training, GPU, Monte Carlo or final-test payload access.","",
              "Validation selected the original weights. Signs are supplied for scoring, not learned. No correction or historical figure reproduction claim."]
    (directory/"RESULTS.md").write_text("\n".join(lines)+"\n")
    _write(directory/"done.json",{"schema_version":1,"plan_sha256":_digest(plan),
                                  "files":{p.name:_sha(p) for p in sorted(directory.iterdir()) if p.is_file()}})
    return directory,summary
