"""Prespecified same-H multi-chain diagnostics for fresh confirmation cohorts.

Independent Hamiltonians are never pooled as R-hat chains. This module keeps
every selected H and flags failures; diagnostic thresholds are empirical checks,
not a certificate of stationarity or a rule for dropping inconvenient instances.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
from statistics import NormalDist
import time

import numpy as np

from quantum_lattice.data.final_confirmation import load_confirmation_plan, _done as _cohort_done
from quantum_lattice.data.state_storage import decode_states
from quantum_lattice.lattice import square_lattice
from quantum_lattice.observables.conditioned import paired_estimators, split_rhat_from_summaries
from quantum_lattice.observables.statistics import autocorrelation_diagnostic, summarize
from quantum_lattice.simulation.estimator_validation import advance_chain, load_chain_records, _initial_chain

VERSION = "fresh-H-multichain-diagnostics-v1"
NAMESPACE = 2026102809


def _read(path):
    return json.loads(Path(path).read_text())


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def _finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def _split(values):
    n = values.shape[1] // 2
    return np.concatenate((values[:, :n], values[:, -n:]), axis=0)


def _rank_normalize(values):
    flat=np.asarray(values).ravel()
    order=np.argsort(flat,kind="mergesort")
    ordered=flat[order]
    starts=np.r_[0,np.flatnonzero(ordered[1:]!=ordered[:-1])+1]
    counts=np.diff(np.r_[starts,len(flat)])
    average_rank=starts+(counts-1)/2+1
    ranks=np.empty(len(flat),dtype=np.float64)
    ranks[order]=np.repeat(average_rank,counts)
    probabilities=(ranks-.375)/(len(flat)+.25)
    normal=NormalDist()
    return np.fromiter((normal.inv_cdf(float(p)) for p in probabilities),
                       dtype=np.float64,count=len(flat)).reshape(values.shape)


def _rhat(split):
    m, n = split.shape
    within = np.var(split, axis=1, ddof=1).mean()
    between = n * np.var(split.mean(axis=1), ddof=1)
    if within == 0:
        return float("inf") if between > 0 else float("nan")
    return float(np.sqrt(((n-1)/n*within + between/n) / within))


def _ess(split):
    """Multi-chain Geyer initial-positive/monotone ESS, capped at draw count.

    Uses between-chain dispersion in the marginal variance estimate. The cap
    is conservative for anticorrelation. Constant/unresolved traces return None.
    """
    m, n = split.shape
    centered = split - split.mean(axis=1, keepdims=True)
    within = np.var(split, axis=1, ddof=1).mean()
    variance = (n-1)/n*within + np.var(split.mean(axis=1), ddof=1)
    if within <= 0 or variance <= 0 or not np.isfinite(variance):
        return None
    spectrum = np.fft.rfft(centered, n=1 << (2*n-1).bit_length(), axis=1)
    covariance = np.fft.irfft(spectrum*spectrum.conj(), axis=1)[:, :n] / n
    rho = 1 - (within - covariance.mean(axis=0)) / variance
    rho[0] = 1
    total, previous = 0., float("inf")
    for lag in range(0,n-1,2):
        pair = float(rho[lag]+rho[lag+1])
        if pair <= 0:
            break
        pair = min(pair, previous)
        total += pair
        previous = pair
    tau = max(1., -1+2*total)
    return float(min(m*n, m*n/tau))


def trace_diagnostics(values, *, block_size_records=45, rhat_threshold=1.01, ess_threshold=400):
    """Rank/folded split-Rhat, rank bulk/tail ESS and actual-trace MCSE.

    Input shape is [independently seeded replicas, chronological retained
    records] for one scalar of one H. MCSE uses actual-value batch means, not
    rank ESS as a substitute. Undefined constant traces remain unresolved.
    """
    a = np.asarray(values)
    if (a.dtype.kind not in "fiu" or a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 4
            or not np.isfinite(a).all() or type(block_size_records) is not int or block_size_records < 1):
        raise ValueError("need >=2 finite real chains, >=4 chronological records and positive block size")
    if (not np.isfinite(rhat_threshold) or rhat_threshold <= 1
            or not np.isfinite(ess_threshold) or ess_threshold <= 0):
        raise ValueError("invalid diagnostic thresholds")
    a = a.astype(np.float64)
    split = _split(a)
    ranked = _rank_normalize(split)
    folded = _rank_normalize(np.abs(split-np.median(split)))
    rank_rhat, folded_rhat = _rhat(ranked), _rhat(folded)
    combined = max(rank_rhat, folded_rhat) if np.isfinite(rank_rhat) and np.isfinite(folded_rhat) else float("nan")
    if np.isinf(rank_rhat) or np.isinf(folded_rhat):
        combined = float("inf")
    half = a.shape[1]//2
    half_means=np.stack((a[:,:half].mean(axis=1),a[:,-half:].mean(axis=1)),axis=1)
    half_variances=np.stack((a[:,:half].var(axis=1,ddof=1),a[:,-half:].var(axis=1,ddof=1)),axis=1)
    conventional = float(split_rhat_from_summaries(half_means,half_variances,half))
    bulk = _ess(ranked)
    low, high = np.quantile(split,[.05,.95])
    tails = [_ess((split <= low).astype(float)), _ess((split >= high).astype(float))]
    tail = min(tails) if all(v is not None for v in tails) else None
    batches = [summarize(chain[:,None],block_size_records) for chain in a]
    block_se = np.array([x["block_se"][0] for x in batches])
    mcse = float(np.sqrt(np.sum(block_se**2))/len(a)) if np.isfinite(block_se).all() else None
    mean = float(a.mean())
    warnings = []
    if not np.isfinite(combined): warnings.append("rank/folded Rhat undefined or infinite; unresolved")
    elif combined > rhat_threshold: warnings.append("rank/folded split Rhat exceeds threshold")
    if bulk is None or bulk < ess_threshold: warnings.append("bulk ESS missing or below threshold")
    if tail is None or tail < ess_threshold: warnings.append("tail ESS missing or below threshold")
    if mcse is None: warnings.append("too few complete blocks for batch-means MCSE")
    return {"replicas":len(a),"retained_records_per_replica":a.shape[1],
            "split_records_per_chain":split.shape[1],"dropped_middle_records_per_replica":a.shape[1]%2,
            "mean":mean,"replica_means":a.mean(axis=1).tolist(),
            "replica_mean_sample_sd":float(a.mean(axis=1).std(ddof=1)),
            "rank_normalized_split_Rhat":_finite_or_none(rank_rhat),
            "folded_rank_normalized_split_Rhat":_finite_or_none(folded_rhat),
            "rank_folded_Rhat":_finite_or_none(combined),
            "conventional_split_Rhat":_finite_or_none(conventional),
            "bulk_ESS":bulk,"tail_ESS":tail,
            "ESS_method":"multi-chain initial-positive monotone sequence; rank bulk/tail; conservative draw-count cap",
            "batch_mean_MCSE":mcse,"block_size_records":block_size_records,
            "complete_blocks_per_replica":a.shape[1]//block_size_records,
            "records_excluded_from_block_means_only":a.shape[1]%block_size_records,
            "relative_MCSE":mcse/abs(mean) if mcse is not None and mean != 0 else None,
            "per_replica_autocorrelation":[autocorrelation_diagnostic(chain) for chain in a],
            "warnings":warnings,"flagged":bool(warnings),
            "scope":"empirical diagnostics for one selected H/quantity; not stationarity or relative-precision certification"}


def _repo(run, repo_root):
    if repo_root is not None:
        return Path(repo_root).resolve()
    for ancestor in Path(run).resolve().parents:
        if (ancestor / "src/quantum_lattice/data/final_confirmation.py").exists():
            return ancestor
    raise ValueError("cannot infer repository root; supply repo_root")


def _source_hashes(root):
    names = ("data/final_diagnostics.py","data/final_confirmation.py","data/state_storage.py",
             "simulation/estimator_validation.py","simulation/solvable.py","observables/conditioned.py",
             "observables/statistics.py","lattice/builders.py","lattice/topology.py","lattice/square.py")
    return {str(Path("src/quantum_lattice")/n):_sha(root/"src/quantum_lattice"/n) for n in names}


def _selected(plan,count):
    selected=[]
    for L in plan["sizes"]:
        candidates=sorted((i for i in plan["instances"] if i["L"]==L),key=lambda i:i["hamiltonian_uid"])
        if len(candidates)<count:
            raise ValueError("diagnostic count exceeds available fresh H at a size")
        selected.extend(candidates[:count])
    return selected


def _fields_and_done(run,cohort,instance):
    marker=_cohort_done(run,cohort,instance)
    if marker is None:
        raise ValueError("selected confirmation instance is incomplete")
    source=run/"instances"/instance["id"]/"records.npz"
    with np.load(source,allow_pickle=False) as a:
        fields=a["fields"]
        if (a["hamiltonian_uid"].item()!=instance["hamiltonian_uid"]
                or fields.shape!=(2*instance["L"]**2,) or not np.isfinite(fields).all()):
            raise ValueError("fresh diagnostic field identity mismatch")
    seeded=np.random.default_rng(np.random.SeedSequence(instance["field_seed"])).uniform(-1.7,1.7,len(fields))
    if not np.array_equal(fields,seeded):
        raise ValueError("fresh diagnostic fields differ from sealed seed")
    return fields,marker


def _plan(root,run,output,cohort,count,replicas,probe_count,rhat_threshold,ess_threshold):
    if type(count) is not int or count<1 or type(replicas) is not int or not 2<=replicas<=16:
        raise ValueError("positive count and2..16replicas required")
    if type(probe_count) is not int or probe_count<1:
        raise ValueError("probe_count must be a positive integer")
    sampling=cohort["sampling"]
    if (sampling["total_sweeps"]-sampling["burn_in_sweeps"])//sampling["thinning_sweeps"]<4:
        raise ValueError("diagnostics require >=4retained records per chain")
    selected=_selected(cohort,count)
    field_pins,jobs,initials={},[],{}
    for instance in selected:
        L=instance["L"]
        fields,done=_fields_and_done(run,cohort,instance)
        field_pins[instance["id"]]=done
        lattice=square_lattice(L)
        seen={hashlib.sha256(np.ones(lattice.n_edges,dtype=np.int8).tobytes()).hexdigest()}
        initials[instance["id"]]=[{"replica":0,"start":"main plus-initialized confirmation chain","physical_state_sha256":next(iter(seen))}]
        for replica in range(1,replicas):
            for nonce in range(1024):
                spec={k:sampling[k] for k in ("backend","lazy_probability","burn_in_sweeps","total_sweeps","thinning_sweeps","chunk_sweeps")}
                spec.update(seed=[cohort["seed"],NAMESPACE,L,instance["index"],replica,nonce],start="random")
                initial=_initial_chain(lattice,fields,spec)
                fingerprint=hashlib.sha256(initial.spins.tobytes()).hexdigest()
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    break
            else:
                raise ValueError("could not construct distinct physical starts")
            initials[instance["id"]].append({"replica":replica,"start":"random valid star-orbit state",
                 "physical_state_sha256":fingerprint,"initial_state_only_duplicate_rejections":nonce})
            jobs.append({"key":f"{instance['id']}/replica-{replica:02d}","instance":instance,"replica":replica,"spec":spec})
    return {"schema_version":1,"version":VERSION,"cohort_run":str(run.relative_to(root)),
        "cohort_plan_sha256":_digest(cohort),"freeze_sha256":cohort["freeze_sha256"],
        "source_sha256":_source_hashes(root),"count_per_size":count,"replicas":replicas,
        "numpy_version":np.__version__,"python_version":platform.python_version(),
        "rank_transform":"NumPy stable average ranks; stdlib NormalDist inverse CDF; Blom offsets",
        "selected_instances":selected,"selection":"first lexicographic UID per size; no outcome/field-strength selection",
        "initializations":initials,"cohort_outputs":field_pins,"jobs":jobs,"probe_count":probe_count,
        "rhat_threshold":rhat_threshold,"ess_threshold":ess_threshold,
        "block_size_records":sampling["block_size_records"],
        "initialization_scope":"domain-separated random valid starts; duplicate physical masks rejected before any diagnostic trajectories; not IID equilibrium starts",
        "scope":"same-H diagnostic only; no selection/filtering/replacement of confirmation H"}


def _chain_done(output,job):
    directory=output/"chains"/job["key"]
    if not (directory/"done.json").exists(): return False
    done=_read(directory/"done.json")
    if done.get("job_sha256")!=_digest(job) or set(done.get("files",{}))!={"progress.json","states_packed.npy"}:
        raise ValueError("diagnostic chain commit identity mismatch")
    if any(_sha(directory/name)!=digest for name,digest in done["files"].items()):
        raise ValueError("diagnostic chain checksum mismatch")
    return True


def _advance_job(output,job,fields):
    directory=Path(output)/"chains"/job["key"]
    if _chain_done(Path(output),job): return True
    complete=advance_chain(directory,square_lattice(job["instance"]["L"]),fields,job["spec"])
    if complete:
        _json(directory/"done.json",{"job_sha256":_digest(job),"files":{name:_sha(directory/name) for name in ("progress.json","states_packed.npy")}})
    return complete


def _worker(job):
    root=Path(_WORKER_OUTPUT)
    fields=np.load(root/"fields"/(job["instance"]["id"]+".npy"),allow_pickle=False)
    return _advance_job(root,job,fields)


def _worker_init(output):
    global _WORKER_OUTPUT
    _WORKER_OUTPUT=output


def _traces(lattice,fields,records,probes):
    names=["log_weight"]+[f"edge_Z_{e}" for e in probes["edge"]]
    for channel in ("star","pair"):
        names += [f"{view}_{channel}_{i}" for view in ("f","g") for i in probes[channel]]
    out=np.empty((len(records),len(names)),dtype=np.float64)
    out[:,0]=records@fields
    out[:,1:1+len(probes["edge"])]=records[:,probes["edge"]]
    for start in range(0,len(records),128):
        values=paired_estimators(lattice,fields,records[start:start+128])
        column=1+len(probes["edge"])
        for channel in ("star","pair"):
            for view in ("f","g"):
                n=len(probes[channel])
                out[start:start+128,column:column+n]=values[channel][view][:,probes[channel]]
                column+=n
    return out,names


def _summarize_H(run,output,plan,instance):
    directory=output/"summaries"/instance["id"]
    if (directory/"done.json").exists():
        done=_read(directory/"done.json")
        if (done.get("plan_sha256")!=_digest(plan) or set(done.get("files",{}))!={"traces.npz","summary.json"}
                or any(_sha(directory/name)!=digest for name,digest in done["files"].items())):
            raise ValueError("diagnostic summary checksum mismatch")
        return _read(directory/"summary.json")
    lattice=square_lattice(instance["L"])
    fields=np.load(output/"fields"/(instance["id"]+".npy"),allow_pickle=False)
    with np.load(run/"instances"/instance["id"]/"records.npz",allow_pickle=False) as a:
        records=[decode_states(a)]
    jobs=[j for j in plan["jobs"] if j["instance"]["id"]==instance["id"]]
    for job in sorted(jobs,key=lambda j:j["replica"]):
        if not _chain_done(output,job): raise ValueError("incomplete diagnostic replicas")
        records.append(load_chain_records(output/"chains"/job["key"],lattice))
    if len({len(r) for r in records})!=1:
        raise ValueError("diagnostic replicas have different retained lengths")
    rng=np.random.default_rng([NAMESPACE,instance["L"],instance["index"],17])
    probes={key:sorted(rng.choice(n,size=min(plan["probe_count"],n),replace=False).tolist())
            for key,n in (("star",lattice.n_vertices),("pair",lattice.n_edges),("edge",lattice.n_edges))}
    values=[_traces(lattice,fields,r,probes) for r in records]
    names=values[0][1]
    traces=np.stack([v[0] for v in values])
    results={name:trace_diagnostics(traces[:,:,i],block_size_records=plan["block_size_records"],
                rhat_threshold=plan["rhat_threshold"],ess_threshold=plan["ess_threshold"]) for i,name in enumerate(names)}
    summary={"instance":instance,"plan_sha256":_digest(plan),"probes":probes,
             "replicas":plan["replicas"],"records_per_replica":len(records[0]),"quantities":results,
             "flagged_quantities":[name for name,result in results.items() if result["flagged"]],
             "cohort_filtering":False,"scope":"selected-H/probe diagnostic; absence of flags is not equilibrium certification"}
    directory.mkdir(parents=True,exist_ok=True)
    temporary=directory/"traces.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle,traces=traces,quantity_names=np.asarray(names,dtype=str),
                            hamiltonian_uid=np.array(instance["hamiltonian_uid"]))
    temporary.replace(directory/"traces.npz")
    _json(directory/"summary.json",summary)
    _json(directory/"done.json",{"plan_sha256":_digest(plan),"files":{name:_sha(directory/name) for name in ("traces.npz","summary.json")}})
    return summary


def run_diagnostics(confirmation_run,output_root,*,repo_root=None,count_per_size=8,replicas=4,
                    workers=5,max_seconds=None,stop_requested=None,probe_count=8,
                    rhat_threshold=1.01,ess_threshold=400):
    """Run a fixed same-H panel, resuming between bounded replica chunks.

    Reuse the main chain as replica0; other replicas start at distinct valid
    random physical configurations with separate proposal RNGs. Operational
    worker/time/pause controls do not change the frozen panel. Already committed
    outputs are verified before reuse. Running this requires completed selected
    fresh-cohort records and the caller's previously sealed model/evaluation plan.
    """
    started=time.monotonic()
    run,output=Path(confirmation_run).resolve(),Path(output_root).resolve()
    root=_repo(run,repo_root)
    allowed=root/"codex material/artifacts/final_benchmark"
    if output==allowed or not output.is_relative_to(allowed):
        raise ValueError("diagnostic output must be within ignored final_benchmark artifacts")
    if type(workers) is not int or workers<1:
        raise ValueError("workers must be a positive integer")
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds<0):
        raise ValueError("max_seconds must be finite and nonnegative")
    cohort=load_confirmation_plan(root,run)
    plan=_plan(root,run,output,cohort,count_per_size,replicas,probe_count,rhat_threshold,ess_threshold)
    output.mkdir(parents=True,exist_ok=True)
    with (output/".diagnostics.lock").open("a") as lock:
        try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError("diagnostics already running") from error
        if (output/"plan.json").exists():
            if _read(output/"plan.json")!=plan or _read(output/"plan-sha256.json")!={"sha256":_digest(plan)}:
                raise ValueError("diagnostic panel/source changed; use new output")
        else:
            if any(p.name!=".diagnostics.lock" for p in output.iterdir()):
                raise ValueError("refusing unbound nonempty diagnostic output")
            _json(output/"plan.json",plan)
            _json(output/"plan-sha256.json",{"sha256":_digest(plan)})
        field_dir=output/"fields"
        field_dir.mkdir(exist_ok=True)
        for instance in plan["selected_instances"]:
            fields,_=_fields_and_done(run,cohort,instance)
            path=field_dir/(instance["id"]+".npy")
            if path.exists():
                if not np.array_equal(np.load(path,allow_pickle=False),fields):
                    raise ValueError("diagnostic fields changed")
            else:
                temporary=path.with_suffix(".npy.tmp")
                with temporary.open("wb") as handle: np.save(handle,fields,allow_pickle=False)
                temporary.replace(path)
        def stopping():
            return ((stop_requested is not None and stop_requested()) or
                    (max_seconds is not None and time.monotonic()-started>=max_seconds))
        pending=[j for j in plan["jobs"] if not _chain_done(output,j)]
        if workers==1:
            while pending and not stopping():
                job=pending.pop(0)
                fields=np.load(field_dir/(job["instance"]["id"]+".npy"),allow_pickle=False)
                if not _advance_job(output,job,fields): pending.append(job)
        elif pending and not stopping():
            context=multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers,mp_context=context,initializer=_worker_init,
                                     initargs=(str(output),)) as executor:
                active={}
                while (pending or active) and not (stopping() and not active):
                    while pending and len(active)<workers and not stopping():
                        job=pending.pop(0)
                        active[executor.submit(_worker,job)]=job
                    if not active: break
                    completed,_=wait(active,timeout=.2,return_when=FIRST_COMPLETED)
                    for future in completed:
                        job=active.pop(future)
                        if not future.result(): pending.append(job)
        summaries=[]
        for instance in plan["selected_instances"]:
            jobs=[j for j in plan["jobs"] if j["instance"]["id"]==instance["id"]]
            committed=(output/"summaries"/instance["id"]/"done.json").exists()
            if all(_chain_done(output,j) for j in jobs) and (committed or not stopping()):
                summaries.append(_summarize_H(run,output,plan,instance))
        completed_jobs=sum(_chain_done(output,j) for j in plan["jobs"])
        complete=len(summaries)==len(plan["selected_instances"])
        report={"schema_version":1,"version":VERSION,"state":"complete" if complete else "paused",
                "plan_sha256":_digest(plan),"selected_H":len(plan["selected_instances"]),
                "completed_additional_chains":completed_jobs,"additional_chains":len(plan["jobs"]),
                "summarized_H":len(summaries),"flagged_H":sum(bool(s["flagged_quantities"]) for s in summaries),
                "current_call_elapsed_seconds":time.monotonic()-started,
                "results":[{"id":s["instance"]["id"],"L":s["instance"]["L"],"uid":s["instance"]["hamiltonian_uid"],
                    "flagged_quantities":s["flagged_quantities"]} for s in summaries],
                "cohort_filtering":False,"scope":"same-H empirical diagnostics; flags retained, not an automatic chain replacement/filter"}
        _json(output/"status.json",report)
        if complete: _json(output/"RESULTS.json",report)
        return report
