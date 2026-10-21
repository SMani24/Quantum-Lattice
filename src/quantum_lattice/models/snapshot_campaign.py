"""Frozen, resumable raw-Z versus complete-edge-mean development pilots.

Scientific settings and record identities are immutable. CUDA is required for
neural research jobs; pseudolikelihood fits run independently on CPU workers.
Preparation and status work without CUDA and never inspect final-test payloads.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
import hashlib
import json
import multiprocessing
from pathlib import Path
import signal
import time

import numpy as np
import torch

from quantum_lattice.data.snapshot_data import prepare_snapshot_cache, load_snapshot_cache
from quantum_lattice.lattice import square_lattice
from quantum_lattice.metrics.hamiltonian_batch import CompiledHamiltonianMetric
from quantum_lattice.models import estimator_mlp as _io
from quantum_lattice.models.inverse_ising import fit_inverse_ising
from quantum_lattice.models.mlp_training import resolve_device
from quantum_lattice.models.snapshot_cnn import snapshot_model_spec
from quantum_lattice.models.snapshot_training import SnapshotTrainSession, _settings

ARTIFACT_ROOT=Path('codex material/artifacts/snapshot_learning/campaigns')
NEURAL_METHODS=('mean_cnn','set_cnn')
_json,_read,_sha,_digest,_torch,_npz,_csv=(_io._json,_io._read,_io._sha,_io._digest,_io._torch,_io._npz,_io._csv)


def _positive(value,name,minimum=1):
    if type(value) is not int or value<minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def validate_config(config):
    c=json.loads(_io._canonical(config))
    expected={'schema_version','L','feature_cache','output_root','selection','seeds','training','inverse_ising','execution','provenance'}
    if not isinstance(c,dict) or set(c)!=expected or type(c['schema_version']) is not int or c['schema_version']!=1 or type(c['L']) is not int or c['L']!=4:
        raise ValueError('Snapshot pilot requires the explicit schema_version=1, L=4 settings')
    for k in ('feature_cache','output_root'):
        if not isinstance(c[k],str) or not c[k] or Path(c[k]).is_absolute():raise ValueError(f'{k} must be repository-relative')
    s=c['selection']
    if not isinstance(s,dict) or set(s)!={'selection_seed','train_hamiltonians','validation_hamiltonians','shot_counts'}:raise ValueError('Invalid snapshot selection settings')
    _positive(s['selection_seed'],'selection_seed',0)
    for k in ('train_hamiltonians','validation_hamiltonians'):_positive(s[k],k)
    if not isinstance(s['shot_counts'],list) or not s['shot_counts'] or s['shot_counts']!=sorted(set(s['shot_counts'])):raise ValueError('shot_counts must be increasing, distinct positive integers')
    for count in s['shot_counts']:_positive(count,'shot_count')
    seeds=c['seeds']
    if not isinstance(seeds,list) or not seeds or len(set(seeds))!=len(seeds):raise ValueError('seeds must be distinct integers')
    for seed in seeds:
        _positive(seed,'seed',0)
        if seed>=2**63:raise ValueError('seed must be below 2**63')
    train=c['training']; required={'device','deterministic','torch_threads','learning_rate','batch_size_hamiltonians','max_epochs','validation_batch_size','early_stopping','shot_chunk_size','activation_checkpoint'}
    if not isinstance(train,dict) or set(train)!=required:raise ValueError('All neural training settings must be explicit')
    _settings({**train,'seed':seeds[0]})
    if train['device']!='cuda' or train['deterministic'] is not True:raise ValueError('Neural research requires deterministic CUDA; no CPU fallback')
    inv=c['inverse_ising']
    if not isinstance(inv,dict) or set(inv)!={'ridge','max_iterations','tolerance'}:raise ValueError('Invalid inverse-Ising settings')
    for name in ('ridge','tolerance'):
        v=inv[name]
        if type(v) not in (int,float) or not np.isfinite(v) or v<=0:raise ValueError(f'Inverse {name} must be positive finite')
    _positive(inv['max_iterations'],'max_iterations')
    execution=c['execution']
    if not isinstance(execution,dict) or set(execution)!={'checkpoint_updates','inverse_workers'}:raise ValueError('Invalid execution settings')
    for k,v in execution.items():_positive(v,k)
    if not isinstance(c['provenance'],dict):raise ValueError('provenance must be an object')
    return c


def _safe_run(root,run):
    allowed=(root/ARTIFACT_ROOT).resolve();path=Path(run).resolve()
    if path==allowed or not path.is_relative_to(allowed):raise ValueError(f'Snapshot runs must be below {allowed}')
    return path


def _sources(root):
    names=['src/quantum_lattice/data/snapshot_data.py','src/quantum_lattice/models/inverse_ising.py','src/quantum_lattice/models/snapshot_cnn.py','src/quantum_lattice/models/snapshot_training.py','src/quantum_lattice/models/snapshot_campaign.py','src/quantum_lattice/models/snapshot_terminal.py','src/quantum_lattice/models/observable_training.py','src/quantum_lattice/models/observable_cnn.py','src/quantum_lattice/metrics/hamiltonian_batch.py','experiments/snapshot_learning/train.py','configs/snapshot_learning.json']
    return {**_io._sources(root),**{name:_sha(root/name) for name in names}}


def _tasks(config):
    result=[]
    for shots in config['selection']['shot_counts']:
        for seed in config['seeds']:
            for method in NEURAL_METHODS:
                result.append({'key':f'L4-M{shots}-seed{seed}-{method}','method':method,'shots':shots,'seed':seed})
        result.append({'key':f'L4-M{shots}-inverse_ising','method':'inverse_ising','shots':shots,'seed':None})
    return result


def _cache_hashes(run):
    cache=run/'cache'
    names=['request.json','manifest.json','train.npz','validation.npz']
    return {name:_sha(cache/name) for name in names}


def prepare(repo_root,config,*,should_stop=None,progress=None):
    """CPU preparation only: resume cache parts then freeze an untrained task list."""
    root=Path(repo_root).resolve();config=validate_config(config)
    run=_safe_run(root,root/config['output_root']);run.mkdir(parents=True,exist_ok=True)
    with _io.campaign_lock(run):
        if (run/'plan.json').exists():
            plan=load_plan(root,run)
            if plan['config']!=config:raise ValueError('Config differs from this frozen run; use a new output_root')
            return run
        feature=(root/config['feature_cache']).resolve()
        if not feature.is_relative_to((root/'codex material/artifacts').resolve()):raise ValueError('Feature provenance must be inside ignored artifacts')
        prepare_snapshot_cache(root,feature,run/'cache',{**config['selection'],'L':config['L']},progress=progress,should_stop=should_stop)
        if should_stop and should_stop():return run
        cache=load_snapshot_cache(run/'cache')
        if cache.L!=config['L'] or len(cache.train)!=config['selection']['train_hamiltonians'] or len(cache.validation)!=config['selection']['validation_hamiltonians'] or list(cache.shot_counts)!=config['selection']['shot_counts']:
            raise ValueError('Prepared data geometry/counts/records differ from the declared pilot')
        plan={'schema_version':1,'workflow':'snapshot_learning','config':config,'tasks':_tasks(config),'source_hashes':_sources(root),'cache_hashes':_cache_hashes(run),'data':{'L':cache.L,'train_hamiltonians':len(cache.train),'validation_hamiltonians':len(cache.validation),'shot_counts':list(cache.shot_counts),'records_per_original_chain':sorted({part['binding']['retained_records'] for split in ('train','validation') for part in cache.metadata['manifest']['parts'][split]}),'target':'abs(b_z)','fields_used_as_inputs':False,'test_payload_loaded':False},'model_specs':{method:snapshot_model_spec(method,shot_chunk_size=config['training']['shot_chunk_size'],activation_checkpoint=config['training']['activation_checkpoint']) for method in NEURAL_METHODS},'protocol':{'scope':'L4 development pilot on reused train/validation Hamiltonian groups; no final test','checkpoint_selection':'strict lowest own-budget validation magnitude MSE; includes epoch0','D5':'secondary reference-sign oracle; inverse native signed D5 separately','same_records':'all methods use identical nested record-index prefixes','independent_unit':'Hamiltonian; record repetitions retained','GPU_binding':'recorded at first neural run; exact device/environment required thereafter','preparation':'no optimizer steps or inverse fits'}}
        _json(run/'plan-sha256.json',{'sha256':_digest(plan)})
        _json(run/'plan.json',plan)
        _report(run,plan)
    return run


def load_plan(repo_root,run):
    root=Path(repo_root).resolve();run=_safe_run(root,run);plan=_read(run/'plan.json')
    if plan.get('workflow')!='snapshot_learning' or plan.get('schema_version')!=1 or _read(run/'plan-sha256.json')!={'sha256':_digest(plan)}:raise ValueError('Invalid frozen snapshot plan')
    config=validate_config(plan['config'])
    if plan['tasks']!=_tasks(config) or plan['source_hashes']!=_sources(root) or plan['cache_hashes']!=_cache_hashes(run):raise ValueError('Snapshot source, task list or data changed; restore the bound version or start a fresh campaign')
    return plan


def _gpu_binding(run):
    device=resolve_device('cuda',True)
    binding={'schema_version':1,'accelerator':_io._accelerator(device),'environment':_io._environment()}
    path=run/'device-binding.json'
    if path.exists() and _read(path)!=binding:raise ValueError('GPU/environment differs from the saved neural run; use a fresh campaign')
    if not path.exists():_json(path,binding)
    return binding


def _job(run,task):
    p=run/'models'/task['key'];p.mkdir(parents=True,exist_ok=True);return p


def _done(job,task,plan):
    path=job/'done.json'
    if not path.exists():return None
    d=_read(path)
    required={'evaluation.json','predictions.npz','per-hamiltonian.csv'}
    required|={'latest.pt','best.pt','history.json'} if task['method'] in NEURAL_METHODS else {'solver-summary.csv'}
    if d.get('task')!=task or d.get('plan_sha256')!=_digest(plan) or set(d.get('hashes',{}))!=required:raise ValueError(f'Invalid completion marker: {task["key"]}')
    for name,digest in d['hashes'].items():
        if _sha(job/name)!=digest:raise ValueError(f'Completed output changed: {task["key"]}/{name}')
    if _read(job/'evaluation.json')!=d.get('summary'):raise ValueError('Completed evaluation summary changed')
    if task['method']=='inverse_ising':
        expected={f'instances/{i:05d}.json' for i in range(plan['data']['validation_hamiltonians'])}
        if set(d.get('instance_hashes',{}))!=expected:raise ValueError('Inverse instance completion inventory changed')
        for name,digest in d['instance_hashes'].items():
            if _sha(job/name)!=digest:raise ValueError('Completed inverse instance changed')
    return d


def _targets(split):
    L=split.L
    return np.abs(split.fields).astype(np.float32).reshape(len(split),L,L,2).transpose(0,3,1,2)


def _inputs(split,task):
    return split.means(task['shots']) if task['method']=='mean_cnn' else split.spin_maps(task['shots'])


def _score(job,task,plan,cache,prediction,*,elapsed,extra=None,native=None):
    split=cache.validation;L=cache.L;E=2*L*L
    prediction=np.asarray(prediction,dtype=np.float32).reshape(len(split),E)
    if not np.isfinite(prediction).all():raise ValueError('Nonfinite predictions')
    target=np.abs(split.fields).astype(np.float32)
    delta=prediction.astype(np.float64)-target.astype(np.float64)
    hmse=np.mean(delta**2,axis=1);hmae=np.mean(np.abs(delta),axis=1)
    recovered=np.sign(split.fields)*prediction
    scores={};native_scores={}
    for model in ('two_channel','z_branch'):
        metric=CompiledHamiltonianMetric(square_lattice(L),model=model)
        scores.update({model+'_'+name:v for name,v in metric.distances(split.fields,recovered).items()})
        if native is not None:native_scores.update({model+'_'+name:v for name,v in metric.distances(split.fields,native).items()})
    summary={'mse':float(hmse.mean()),'rmse':float(np.sqrt(hmse.mean())),'mae':float(hmae.mean()),'mean_H_rmse':float(np.sqrt(hmse).mean()),'negative_prediction_fraction':float(np.mean(prediction<0)),'above_1_7_prediction_fraction':float(np.mean(prediction>1.7)),'reference_sign_D5':{k:float(v.mean()) for k,v in scores.items()}}
    if native is not None:summary['native_signed_D5']={k:float(v.mean()) for k,v in native_scores.items()}
    evaluation={'task':task,'scope':'development pilot validation; no final test','target':'abs(b_z)','prediction_policy':'raw unclipped neural magnitudes; inverse absolute fitted signed coupling','checkpoint_selection':'lowest own-budget validation magnitude MSE' if task['method'] in NEURAL_METHODS else 'per-instance pseudolikelihood solve, no validation-based parameter selection','metrics':summary,'train_hamiltonians':len(cache.train),'validation_hamiltonians':len(split),'elapsed_seconds':elapsed,'elapsed_scope':'neural optimizer advances plus epoch evaluation, excluding I/O/exports; inverse observed parent scheduling time','test_payload_loaded':False,'fields_used_as_inputs':False,**(extra or {})}
    rows=[]
    for i,name in enumerate(split.ids):
        row={'hamiltonian_id':str(name),'hamiltonian_uid':str(split.uids[i]),'mse':float(hmse[i]),'rmse':float(np.sqrt(hmse[i])),'mae':float(hmae[i])}
        row.update({'reference_sign_'+k:float(v[i]) for k,v in scores.items()});row.update({'native_signed_'+k:float(v[i]) for k,v in native_scores.items()});rows.append(row)
    _csv(job/'per-hamiltonian.csv',rows,tuple(rows[0]))
    arrays={'prediction':prediction,'target':target,'hamiltonian_ids':split.ids,'hamiltonian_uids':split.uids,'record_indices':split.record_indices[:,:task['shots']]}
    if native is not None:arrays['native_signed_fields']=np.asarray(native,dtype=np.float64)
    _npz(job/'predictions.npz',**arrays);_json(job/'evaluation.json',evaluation)
    return evaluation


def _complete(job,task,plan,summary):
    names=['evaluation.json','predictions.npz','per-hamiltonian.csv']
    names+=['latest.pt','best.pt','history.json'] if task['method'] in NEURAL_METHODS else ['solver-summary.csv']
    marker={'schema_version':1,'task':task,'plan_sha256':_digest(plan),'hashes':{n:_sha(job/n) for n in names},'summary':summary}
    if task['method']=='inverse_ising':marker['instance_hashes']={f'instances/{i:05d}.json':_sha(job/f'instances/{i:05d}.json') for i in range(plan['data']['validation_hamiltonians'])}
    _json(job/'done.json',marker)


def _run_neural(run,plan,cache,task,stop,progress):
    binding=_gpu_binding(run);job=_job(run,task);payload=None
    if (job/'latest.pt').exists():
        payload=torch.load(job/'latest.pt',map_location='cpu',weights_only=False)
        if payload.get('task')!=task or payload.get('plan_sha256')!=_digest(plan) or payload.get('device_binding_sha256')!=_digest(binding):raise ValueError('Neural checkpoint task/plan/device mismatch')
    session=SnapshotTrainSession(_inputs(cache.train,task),_targets(cache.train),_inputs(cache.validation,task),_targets(cache.validation),{**plan['config']['training'],'seed':task['seed']},method=task['method'],checkpoint=payload['session'] if payload else None)
    elapsed=float(payload['elapsed_seconds']) if payload else 0.
    def save():
        _torch(job/'latest.pt',{'schema_version':1,'task':task,'plan_sha256':_digest(plan),'device_binding_sha256':_digest(binding),'elapsed_seconds':elapsed,'session':session.snapshot()})
    if payload is None:save()
    while not session.finished and not stop():
        started=time.perf_counter();session.advance(plan['config']['execution']['checkpoint_updates'])
        if session.device.type=='cuda':torch.cuda.synchronize(session.device)
        elapsed+=time.perf_counter()-started;save()
        if progress:progress({'stage':'training','task':task['key'],'epoch':session.epoch,'step':session.step,'best_mse':session.best_val_mse})
    if session.finished:
        prediction=session.selected_predictions(_inputs(cache.validation,task))
        canonical=prediction.transpose(0,2,3,1).reshape(len(cache.validation),-1)
        summary=_score(job,task,plan,cache,canonical,elapsed=elapsed,extra={'best_epoch':session.best_epoch,'best_step':session.best_step,'best_validation_mse':session.best_val_mse,'last_epoch':session.epoch,'last_step':session.step,'stop_reason':session.stop_reason,'model_spec':plan['model_specs'][task['method']]})
        _json(job/'history.json',session.history)
        _torch(job/'best.pt',{'schema_version':1,'task':task,'plan_sha256':_digest(plan),'device_binding_sha256':_digest(binding),'best_epoch':session.best_epoch,'best_step':session.best_step,'model_state_dict':session.best_state_dict,'model_spec':plan['model_specs'][task['method']]})
        _complete(job,task,plan,summary)
    del session


def _worker_init():
    signal.signal(signal.SIGINT,signal.SIG_IGN)


def _inverse_worker(states,settings):
    return fit_inverse_ising(states,**settings)


def _inverse_result(path,task,uid,plan):
    if not path.exists():return None
    value=_read(path)
    if value.get('task')!=task or value.get('uid')!=str(uid) or value.get('plan_sha256')!=_digest(plan):raise ValueError('Inverse partial result identity changed')
    result=value['result']
    if value.get('result_sha256')!=_digest(result):raise ValueError('Saved inverse fit checksum changed')
    fields=np.asarray(result['fields'],dtype=np.float64)
    if fields.shape!=(32,) or not np.isfinite(fields).all() or result['stop_reason']=='paused':raise ValueError('Invalid committed inverse result')
    return result


def _run_inverse(run,plan,cache,task,stop,progress,workers):
    job=_job(run,task);parts=job/'instances';parts.mkdir(exist_ok=True)
    split=cache.validation;results={};pending=[]
    for i,uid in enumerate(split.uids):
        value=_inverse_result(parts/f'{i:05d}.json',task,uid,plan)
        if value is None:pending.append(i)
        else:results[i]=value
    elapsed_path=job/'elapsed.json';elapsed=float(_read(elapsed_path)['seconds']) if elapsed_path.exists() else 0.
    start=time.perf_counter()
    def save(i,result):
        _json(parts/f'{i:05d}.json',{'task':task,'uid':str(split.uids[i]),'plan_sha256':_digest(plan),'result':result,'result_sha256':_digest(result)})
        results[i]=result
        _json(job/'progress.json',{'completed':len(results),'total':len(split),'workers':workers})
        _json(elapsed_path,{'seconds':elapsed+time.perf_counter()-start})
        if progress:progress({'stage':'inverse','task':task['key'],'completed':len(results),'total':len(split)})
    if workers==1:
        for i in pending:
            if stop():break
            save(i,_inverse_worker(split.states[i,:task['shots']],plan['config']['inverse_ising']))
    elif pending and not stop():
        iterator=iter(pending);active={}
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=_worker_init) as executor:
            def fill():
                while len(active)<workers and not stop():
                    i=next(iterator,None)
                    if i is None:break
                    future=executor.submit(_inverse_worker,split.states[i,:task['shots']],plan['config']['inverse_ising']);active[future]=i
            fill()
            while active:
                finished,_=wait(active,timeout=.5,return_when=FIRST_COMPLETED)
                for future in finished:save(active.pop(future),future.result())
                fill()
    elapsed+=time.perf_counter()-start;_json(elapsed_path,{'seconds':elapsed})
    if len(results)==len(split):
        signed=np.stack([results[i]['fields'] for i in range(len(split))])
        summary=_score(job,task,plan,cache,np.abs(signed),elapsed=elapsed,native=signed,extra={'solver_settings':plan['config']['inverse_ising'],'converged_instances':sum(bool(r['converged']) for r in results.values()),'total_instances':len(split),'status':'all attempted; convergence flags retained','workers_last_session':workers})
        rows=[{'hamiltonian_id':str(split.ids[i]),'hamiltonian_uid':str(split.uids[i]),'converged':results[i]['converged'],'stop_reason':results[i]['stop_reason'],'iteration':results[i]['iteration'],'objective':results[i]['objective'],'gradient_norm':results[i]['gradient_norm']} for i in range(len(split))]
        _csv(job/'solver-summary.csv',rows,tuple(rows[0]));_complete(job,task,plan,summary)


def _report(run,plan,*,persist=True):
    rows=[]
    for task in plan['tasks']:
        job=run/'models'/task['key'];done=_done(job,task,plan)
        row={**task,'state':'complete' if done else 'pending'}
        if done:row.update(done['summary'])
        elif (job/'latest.pt').exists():
            payload=torch.load(job/'latest.pt',map_location='cpu',weights_only=False)
            if payload.get('task')!=task or payload.get('plan_sha256')!=_digest(plan):raise ValueError('Saved progress identity mismatch')
            inner=payload['session'].get('engine',payload['session'])
            row.update(state='paused',epoch=inner.get('epoch'),step=inner.get('step'))
        elif (job/'progress.json').exists():row.update(state='paused',inverse_progress=_read(job/'progress.json'))
        rows.append(row)
    status={'scope':'snapshot development pilot; no final test','completed':sum(r['state']=='complete' for r in rows),'total':len(rows),'tasks':rows,'test_payload_loaded':False}
    if not persist:return status
    _json(run/'status.json',status)
    flat=[{'method':r['method'],'shots':r['shots'],'seed':r['seed'],'mse':r['metrics']['mse'],'rmse':r['metrics']['rmse'],'mae':r['metrics']['mae'],'D5_reference_sign':r['metrics']['reference_sign_D5']['two_channel_full'],'elapsed_seconds':r['elapsed_seconds'],'stop_reason':r.get('stop_reason','per-instance solver'),'converged_instances':r.get('converged_instances')} for r in rows if r['state']=='complete']
    if flat:_csv(run/'comparison.csv',flat,tuple(flat[0]))
    text='# Raw-configuration pilot\n\nDevelopment only. All methods share Hamiltonians and exact nested record subsets. Neural models use CUDA; inverse fits use CPU.\n\n'
    text+=f"Complete: {status['completed']}/{status['total']}. No final-test payloads.\n\n"
    text+='| Method | Records/H | Seed | State | Validation MSE |\n|---|---:|---:|---|---:|\n'
    for r in rows:text+=f"| {r['method']} | {r['shots']} | {r['seed'] if r['seed'] is not None else '—'} | {r['state']} | {r['metrics']['mse'] if r['state']=='complete' else '—'} |\n"
    text+='\nOnly compare pilot methods within the same record budget. Old full-data observable-map runs are contextual. See the experiment README for architecture/protocol and pause/resume.\n\nFiles: plan.json and source/data hashes; cache/ raw bags; device-binding.json at first CUDA run; models/<task>/ checkpoints/predictions/evaluation/per-H errors; comparison.csv; inverse instances and solver-summary.csv.\n'
    (run/'RESULTS.md').write_text(text)
    return status


def progress(repo_root,config):
    root=Path(repo_root).resolve();config=validate_config(config);run=_safe_run(root,root/config['output_root'])
    if not (run/'plan.json').exists():
        cache=run/'cache/progress.json'
        return {'state':'preparing' if cache.exists() else 'not_prepared','data_progress':_read(cache) if cache.exists() else None,'completed':0,'total':len(_tasks(config)),'tasks':[]}
    plan=load_plan(root,run)
    if config!=plan['config']:raise ValueError('Config differs from this frozen run')
    return _report(run,plan,persist=False)


def run_campaign(repo_root,config,*,workers=None,progress_callback=None):
    """Resume data and each task. Ctrl+C pauses safely; never discards a job."""
    root=Path(repo_root).resolve();config=validate_config(config)
    workers=_positive(workers if workers is not None else config['execution']['inverse_workers'],'workers')
    requested={'stop':False};old=signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT,lambda *_:requested.update(stop=True))
    stop=lambda:requested['stop']
    try:
        run=prepare(root,config,should_stop=stop,progress=progress_callback)
        if stop() or not (run/'plan.json').exists():return progress(root,config)
        with _io.campaign_lock(run):
            plan=load_plan(root,run);cache=load_snapshot_cache(run/'cache')
            for task in plan['tasks']:
                if stop():break
                job=_job(run,task)
                if _done(job,task,plan):continue
                if task['method'] in NEURAL_METHODS:_run_neural(run,plan,cache,task,stop,progress_callback)
                else:_run_inverse(run,plan,cache,task,stop,progress_callback,workers)
                _report(run,plan)
            return _report(run,plan)
    finally:signal.signal(signal.SIGINT,old)
