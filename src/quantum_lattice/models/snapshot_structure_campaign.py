"""Matched residual/gauge factorial on immutable raw-snapshot development bags.

This separate workflow reads the prepared scaling cache without changing it.
Known fields are labels/scoring references only. No new Monte Carlo, inverse fit
or final-test payload is needed; all long neural runs are user launched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from quantum_lattice.data.snapshot_gauge import gauge_mask_spec, vertex_gauge_masks
from quantum_lattice.models import snapshot_campaign as pilot
from quantum_lattice.models import snapshot_scaling as scaling
from quantum_lattice.models.snapshot_structure import structure_model_spec
from quantum_lattice.models.snapshot_structure_training import StructureTrainSession

ARTIFACT_ROOT = Path('codex material/artifacts/snapshot_structure/campaigns')
NEURAL_METHODS = pilot.NEURAL_METHODS
_json, _read, _sha, _digest, _csv = pilot._json, pilot._read, pilot._sha, pilot._digest, pilot._csv
_DIAGNOSTIC_FILES = ('gauge-summary.json', 'gauge-diagnostics.npz')


def validate_config(config):
    """Validate all scientific factors and reuse the frozen training validator."""
    value = json.loads(pilot._io._canonical(config))
    expected = {'schema_version', 'L', 'source_campaign', 'output_root', 'selection',
                'seeds', 'factors', 'training', 'inverse_ising', 'inverse_reference',
                'execution', 'diagnostics', 'provenance'}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError('Invalid snapshot structural settings')
    selection = value['selection']
    if not isinstance(selection, dict) or set(selection) != {
            'selection_seed', 'train_hamiltonians', 'validation_hamiltonians', 'shots'}:
        raise ValueError('Structural selection requires one training count and record budget')
    pilot._positive(selection['train_hamiltonians'], 'train_hamiltonians')
    factors = value['factors']
    if not isinstance(factors, dict) or set(factors) != {'residual', 'gauge_augmentation'}:
        raise ValueError('Declare residual and gauge_augmentation factors explicitly')
    for name, levels in factors.items():
        if (not isinstance(levels, list) or any(type(v) is not bool for v in levels)
                or levels not in ([False], [False, True])):
            raise ValueError(f'{name} must be [false] or [false,true], retaining plain controls')
    for name in ('source_campaign', 'output_root'):
        scaling._relative(value[name], name)
    diagnostics = value['diagnostics']
    if not isinstance(diagnostics, dict) or set(diagnostics) != {'gauge_masks', 'gauge_seed'}:
        raise ValueError('Declare gauge_masks and gauge_seed diagnostics')
    pilot._positive(diagnostics['gauge_masks'], 'gauge_masks')
    if diagnostics['gauge_masks'] > 64:
        raise ValueError('gauge_masks must be at most64 for bounded diagnostics')
    pilot._positive(diagnostics['gauge_seed'], 'gauge_seed', 0)
    if diagnostics['gauge_seed'] >= 2**63:
        raise ValueError('gauge_seed must be below2**63')
    adapted = {k: v for k, v in value.items()
               if k not in {'source_campaign', 'factors', 'diagnostics'}}
    adapted['feature_cache'] = value['source_campaign'] + '/cache'
    adapted['selection'] = {**selection, 'train_hamiltonian_counts':
                            [selection['train_hamiltonians']]}
    adapted['selection'].pop('train_hamiltonians')
    scaling.validate_config(adapted)
    return value


def _safe_run(root, run):
    allowed = (root / ARTIFACT_ROOT).resolve()
    path = Path(run).resolve()
    if path == allowed or not path.is_relative_to(allowed):
        raise ValueError(f'Structural runs must be below {allowed}')
    return path


def _sources(root):
    names = ('src/quantum_lattice/data/snapshot_gauge.py',
             'src/quantum_lattice/models/snapshot_structure.py',
             'src/quantum_lattice/models/snapshot_structure_training.py',
             'src/quantum_lattice/models/snapshot_structure_campaign.py',
             'src/quantum_lattice/models/snapshot_structure_terminal.py',
             'experiments/snapshot_structure/train.py', 'configs/snapshot_structure.json')
    return {**scaling._sources(root), **{name: _sha(root / name) for name in names}}


def _tasks(config):
    selection = config['selection']
    tasks = []
    for seed in config['seeds']:
        for method in NEURAL_METHODS:
            for residual in config['factors']['residual']:
                for gauge in config['factors']['gauge_augmentation']:
                    arm = ('R' if residual else 'P') + ('G' if gauge else '0')
                    tasks.append({
                        'key': f"L4-H{selection['train_hamiltonians']}-M{selection['shots']}-seed{seed}-{method}-{arm}",
                        'method': method, 'arm': arm, 'residual': residual,
                        'gauge_augmentation': gauge, 'shots': selection['shots'],
                        'seed': seed, 'train_hamiltonians': selection['train_hamiltonians']})
    return tasks


def _spec_key(task):
    return task['method'] + ('_residual' if task['residual'] else '_plain')


def _specs(config):
    return {method + ('_residual' if residual else '_plain'): structure_model_spec(
                method, residual=residual, shot_chunk_size=config['training']['shot_chunk_size'],
                activation_checkpoint=config['training']['activation_checkpoint'])
            for method in NEURAL_METHODS for residual in config['factors']['residual']}


def _data_reference(root, config, *, expected=None, load=False):
    """Bind only immutable plan/cache files; old model progress may keep changing."""
    source = scaling._safe_run(root, root / config['source_campaign'])
    source_plan = scaling.load_plan(root, source)
    selection = config['selection']
    original = source_plan['config']['selection']
    if (source_plan['config']['L'] != config['L']
            or original['selection_seed'] != selection['selection_seed']
            or original['shots'] != selection['shots']
            or original['validation_hamiltonians'] != selection['validation_hamiltonians']
            or max(original['train_hamiltonian_counts']) < selection['train_hamiltonians']):
        raise ValueError('Structural data selection differs from frozen scaling data')
    binding = {'campaign_root': str(source.relative_to(root)),
               'plan_sha256': _digest(source_plan), 'cache_hashes': source_plan['cache_hashes'],
               'training_prefix': selection['train_hamiltonians'],
               'selection_seed': selection['selection_seed'], 'shots': selection['shots']}
    if expected is not None and binding != expected:
        raise ValueError('Frozen structural data reference changed')
    if not load:
        return binding, None
    cache = pilot.load_snapshot_cache(source / 'cache')
    cache = scaling.prefix_cache(cache, selection['train_hamiltonians'])
    if cache.L != config['L'] or len(cache.validation) != selection['validation_hamiltonians']:
        raise ValueError('Structural cache geometry/counts differ')
    return binding, cache


def prepare(repo_root, config, *, should_stop=None, progress=None):
    """Freeze a fresh queue and read-only data reference; no training or fitting."""
    root = Path(repo_root).resolve()
    config = validate_config(config)
    run = _safe_run(root, root / config['output_root'])
    run.mkdir(parents=True, exist_ok=True)
    with pilot._io.campaign_lock(run):
        if (run / 'plan.json').exists():
            plan = load_plan(root, run)
            if plan['config'] != config:
                raise ValueError('Config differs from frozen structural run; use a fresh output_root')
            return run
        if should_stop and should_stop():
            return run
        if progress:
            progress({'stage': 'preparing', 'completed': 0, 'total': 1})
        binding, cache = _data_reference(root, config, load=True)
        if should_stop and should_stop():
            return run
        reference = scaling._reference(root, config, cache)
        steps = (len(cache.train) + config['training']['batch_size_hamiltonians'] - 1) // config['training']['batch_size_hamiltonians']
        plan = {
            'schema_version': 1, 'workflow': 'snapshot_structure', 'config': config,
            'tasks': _tasks(config), 'source_hashes': _sources(root),
            'data_reference': binding, 'inverse_reference': reference,
            'model_specs': _specs(config), 'updates_per_epoch': steps,
            'max_optimizer_updates': steps * config['training']['max_epochs'],
            'data': {'L': cache.L, 'train_hamiltonians': len(cache.train),
                     'validation_hamiltonians': len(cache.validation),
                     'shots': config['selection']['shots'], 'target': 'abs(b_z)',
                     'train_uids_sha256': _digest(cache.train.uids.tolist()),
                     'validation_uids_sha256': _digest(cache.validation.uids.tolist()),
                     'fields_used_as_inputs': False, 'test_payload_loaded': False},
            'protocol': {
                'scope': 'L4 residual/gauge factorial; reused development validation; no final test',
                'fresh_fits': 'All seeds/arms train from scratch, including plain controls',
                'augmentation': 'One keyed root-fixed vertex mask per UID/epoch, shared over whole bag; seed paired across arms',
                'selection': 'Strict best original-view validation MSE includingepoch0; no gauge-view checkpoint selection',
                'information': 'True fields are labels/scoring only; augmentation needs no true fields',
                'stationarity': 'Equilibrium gauge identity does not remove finite-initialization bias',
                'D5': 'Reference-sign assistance explicit; inverse native signs separate',
                'inverse': 'One immutable matched reference, never refitted or seed replicated',
                'uncertainty': 'Hamiltonian is statistical unit; records/edges/masks/seeds not independent H',
                'GPU_binding': 'Fresh first neural run binds device/environment; CUDA required',
                'preparation': 'Zero optimizer updates, new inverse fits or new sampling'}}
        _json(run / 'data-reference.json', binding)
        _json(run / 'inverse-reference.json', reference)
        _json(run / 'plan-sha256.json', {'sha256': _digest(plan)})
        _json(run / 'plan.json', plan)
        _report(run, plan)
        if progress:
            progress({'stage': 'preparing', 'completed': 1, 'total': 1})
    return run


def load_plan(repo_root, run):
    root = Path(repo_root).resolve()
    run = _safe_run(root, run)
    plan = _read(run / 'plan.json')
    if (plan.get('schema_version') != 1 or plan.get('workflow') != 'snapshot_structure'
            or _read(run / 'plan-sha256.json') != {'sha256': _digest(plan)}):
        raise ValueError('Invalid frozen structural plan')
    config = validate_config(plan['config'])
    if (plan['tasks'] != _tasks(config) or plan['source_hashes'] != _sources(root)
            or plan['model_specs'] != _specs(config)):
        raise ValueError('Structural source/tasks/models changed; restore or use a fresh campaign')
    if _read(run / 'data-reference.json') != plan['data_reference']:
        raise ValueError('Structural data reference binding changed')
    _data_reference(root, config, expected=plan['data_reference'])
    if _read(run / 'inverse-reference.json') != plan['inverse_reference']:
        raise ValueError('Structural inverse reference binding changed')
    scaling._verify_reference(root, plan['inverse_reference'])
    return plan


def _done(job, task, plan):
    marker = pilot._done(job, task, plan)
    if marker is None:
        return None
    hashes = marker.get('diagnostic_hashes', {})
    if set(hashes) != set(_DIAGNOSTIC_FILES):
        raise ValueError('Structural diagnostic completion inventory changed')
    for name, expected in hashes.items():
        if _sha(job / name) != expected:
            raise ValueError('Completed structural gauge diagnostic changed')
    return marker


def _check_checkpoint(run, task, plan, payload):
    scaling._check_progress_checkpoint(run, task, plan, payload)
    session = payload.get('session', {})
    uids = session.get('train_uids')
    if (not isinstance(uids, list) or len(uids) != plan['data']['train_hamiltonians']
            or any(not isinstance(uid, str) or not uid.strip() for uid in uids)
            or len(set(uids)) != len(uids)):
        raise ValueError('Structural checkpoint Hamiltonian identities changed')
    encoded_uids = json.dumps(uids, ensure_ascii=True, separators=(',', ':')).encode('utf8')
    expected_augmentation = {
        'mask_spec': gauge_mask_spec(), 'enabled': task['gauge_augmentation'],
        'seed': task['seed'], 'epoch_index': 'zero_based_optimizer_epoch',
        'scope': 'whole_hamiltonian_bag', 'validation': 'unaugmented',
        'train_uids_sha256': hashlib.sha256(encoded_uids).hexdigest()}
    if (session.get('method') != task['method']
            or session.get('training_kind') != 'snapshot_structure_cnn'
            or type(session.get('training_schema_version')) is not int
            or session.get('training_schema_version') != 1
            or type(session.get('residual')) is not bool
            or session.get('residual') != task['residual']
            or type(session.get('gauge_augmentation')) is not bool
            or session.get('gauge_augmentation') != task['gauge_augmentation']
            or _digest(uids) != plan['data']['train_uids_sha256']
            or session.get('augmentation_spec') != expected_augmentation
            or session.get('model_spec') != plan['model_specs'][_spec_key(task)]):
        raise ValueError('Structural checkpoint model binding changed')
    expected_config, _ = pilot._settings({**plan['config']['training'], 'seed': task['seed']})
    engine = session.get('engine', {})
    elapsed = payload.get('elapsed_seconds')
    if (not isinstance(engine, dict) or engine.get('config') != expected_config
            or type(elapsed) not in (float, int) or not np.isfinite(elapsed) or elapsed < 0):
        raise ValueError('Structural checkpoint training settings or elapsed time changed')


def _gauge_diagnostics(session, cache, task, config, base, stop, progress):
    inputs = pilot._inputs(cache.validation, task)
    target = pilot._targets(cache.validation).astype(np.float64)
    count = config['diagnostics']['gauge_masks']
    disagreement, error = [], []
    for index in range(count):
        if stop():
            return None
        masks = vertex_gauge_masks(cache.validation.uids.tolist(), cache.L,
                                   seed=config['diagnostics']['gauge_seed'], epoch=index)
        masked = inputs * (masks if task['method'] == 'mean_cnn' else masks[:, None])
        prediction = session.selected_predictions(masked)
        disagreement.append(np.mean((prediction.astype(np.float64) - base.astype(np.float64))**2, axis=(1, 2, 3)))
        error.append(np.mean((prediction.astype(np.float64) - target)**2, axis=(1, 2, 3)))
        if progress:
            progress({'stage': 'diagnostics', 'task': task['key'],
                      'completed': index + 1, 'total': count})
    return {'squared_prediction_difference': np.stack(disagreement),
            'transformed_view_mse': np.stack(error)}


def _run_neural(run, plan, cache, task, stop, progress):
    binding = pilot._gpu_binding(run)
    job = pilot._job(run, task)
    payload = None
    if (job / 'latest.pt').exists():
        payload = torch.load(job / 'latest.pt', map_location='cpu', weights_only=False)
        _check_checkpoint(run, task, plan, payload)
    session = StructureTrainSession(
        pilot._inputs(cache.train, task), pilot._targets(cache.train),
        pilot._inputs(cache.validation, task), pilot._targets(cache.validation),
        {**plan['config']['training'], 'seed': task['seed']}, method=task['method'],
        residual=task['residual'], gauge_augmentation=task['gauge_augmentation'],
        train_uids=cache.train.uids.tolist(), checkpoint=payload['session'] if payload else None)
    elapsed = float(payload['elapsed_seconds']) if payload else 0.0
    def save():
        pilot._torch(job / 'latest.pt', {
            'schema_version': 1, 'task': task, 'plan_sha256': _digest(plan),
            'device_binding_sha256': _digest(binding), 'elapsed_seconds': elapsed,
            'session': session.snapshot()})
    if payload is None:
        save()
    while not session.finished and not stop():
        started = time.perf_counter()
        session.advance(plan['config']['execution']['checkpoint_updates'])
        if session.device.type == 'cuda':
            torch.cuda.synchronize(session.device)
        elapsed += time.perf_counter() - started
        save()
        if progress:
            progress({'stage': 'training', 'task': task['key'], 'epoch': session.epoch,
                      'step': session.step, 'best_mse': session.best_val_mse})
    if session.finished and not stop():
        prediction = session.selected_predictions(pilot._inputs(cache.validation, task))
        started = time.perf_counter()
        diagnostics = _gauge_diagnostics(session, cache, task, plan['config'], prediction, stop, progress)
        if diagnostics is None:
            return
        diagnostic_seconds = time.perf_counter() - started
        summary = {'gauge_prediction_mse': float(diagnostics['squared_prediction_difference'].mean()),
                   'transformed_view_mse': float(diagnostics['transformed_view_mse'].mean()),
                   'masks_per_hamiltonian': plan['config']['diagnostics']['gauge_masks'],
                   'seed': plan['config']['diagnostics']['gauge_seed'],
                   'elapsed_seconds': diagnostic_seconds,
                   'scope': 'Post-selection diagnostic; same H under gauge views, not independent H; no group-averaged primary prediction'}
        pilot._npz(job / 'gauge-diagnostics.npz', **diagnostics,
                   hamiltonian_uids=cache.validation.uids)
        _json(job / 'gauge-summary.json', summary)
        canonical = prediction.transpose(0, 2, 3, 1).reshape(len(cache.validation), -1)
        evaluation = pilot._score(job, task, plan, cache, canonical, elapsed=elapsed, extra={
            'scope': 'Structural development validation; no final test',
            'best_epoch': session.best_epoch, 'best_step': session.best_step,
            'best_validation_mse': session.best_val_mse, 'last_epoch': session.epoch,
            'last_step': session.step, 'stop_reason': session.stop_reason,
            'model_spec': plan['model_specs'][_spec_key(task)], 'gauge_diagnostics': summary})
        _json(job / 'history.json', session.history)
        pilot._torch(job / 'best.pt', {
            'schema_version': 1, 'task': task, 'plan_sha256': _digest(plan),
            'device_binding_sha256': _digest(binding), 'best_epoch': session.best_epoch,
            'best_step': session.best_step, 'model_state_dict': session.best_state_dict,
            'model_spec': plan['model_specs'][_spec_key(task)]})
        names = ('evaluation.json', 'predictions.npz', 'per-hamiltonian.csv',
                 'latest.pt', 'best.pt', 'history.json')
        _json(job / 'done.json', {
            'schema_version': 1, 'task': task, 'plan_sha256': _digest(plan),
            'hashes': {name: _sha(job / name) for name in names},
            'diagnostic_hashes': {name: _sha(job / name) for name in _DIAGNOSTIC_FILES},
            'summary': evaluation})
    del session


def factorial_contrasts(rows):
    """Complete matched four-arm seed contrasts, without pooling H as seeds."""
    groups = {}
    for row in rows:
        if row.get('state') == 'complete':
            groups.setdefault((row['method'], row['seed']), {})[row['arm']] = row
    result = []
    for (method, seed), arms in sorted(groups.items()):
        if set(arms) != {'P0', 'PG', 'R0', 'RG'}:
            continue
        losses = {arm: row['metrics']['mse'] for arm, row in arms.items()}
        a, b, c, d = (losses[k] for k in ('P0', 'PG', 'R0', 'RG'))
        result.append({'method': method, 'seed': seed,
                       'residual_main_effect_mse': ((c-a)+(d-b))/2,
                       'gauge_main_effect_mse': ((b-a)+(d-c))/2,
                       'interaction_mse': d-c-b+a,
                       'scope': 'Development contrast; negative main effects favor lower MSE; negative interaction is superadditive, not necessarily overall improvement; not a significance test'})
    return result


def _report(run, plan, *, persist=True):
    rows = []
    for task in plan['tasks']:
        job = run / 'models' / task['key']
        done = _done(job, task, plan)
        row = {**task, 'state': 'complete' if done else 'pending'}
        if done:
            payload = torch.load(job / 'latest.pt', map_location='cpu', weights_only=False)
            _check_checkpoint(run, task, plan, payload)
            row.update(done['summary'])
        elif (job / 'latest.pt').exists():
            payload = torch.load(job / 'latest.pt', map_location='cpu', weights_only=False)
            _check_checkpoint(run, task, plan, payload)
            engine = payload['session']['engine']
            row.update(state='paused', epoch=engine['epoch'], step=engine['step'])
        rows.append(row)
    cells = []
    for method in NEURAL_METHODS:
        for residual in plan['config']['factors']['residual']:
            for gauge in plan['config']['factors']['gauge_augmentation']:
                selected = [r for r in rows if r['method'] == method and r['residual'] == residual
                            and r['gauge_augmentation'] == gauge and r['state'] == 'complete']
                cells.append({'method': method, 'residual': residual, 'gauge_augmentation': gauge,
                              'completed_seeds': len(selected), 'total_seeds': len(plan['config']['seeds']),
                              'mse': float(np.mean([r['metrics']['mse'] for r in selected])) if selected else None})
    reference = plan['inverse_reference']
    status = {'scope': plan['protocol']['scope'], 'completed': sum(r['state'] == 'complete' for r in rows),
              'total': len(rows), 'tasks': rows, 'cells': cells,
              'train_hamiltonians': plan['data']['train_hamiltonians'],
              'shots': plan['data']['shots'], 'test_payload_loaded': False,
              'inverse_reference': {'state': 'complete', 'reused_without_refitting': True,
                                    'metrics': reference['summary']['metrics'],
                                    'source': reference['campaign_root']}}
    if not persist:
        return status
    _json(run / 'status.json', status)
    flat = [{'method': r['method'], 'arm': r['arm'], 'residual': r['residual'],
             'gauge_augmentation': r['gauge_augmentation'], 'seed': r['seed'],
             'train_hamiltonians': r['train_hamiltonians'], 'shots': r['shots'],
             'mse': r['metrics']['mse'],
             'D5_reference_sign': r['metrics']['reference_sign_D5']['two_channel_full'],
             'gauge_prediction_mse': r['gauge_diagnostics']['gauge_prediction_mse'],
             'elapsed_seconds': r['elapsed_seconds'], 'stop_reason': r['stop_reason'],
             'provenance': 'fresh structural fit'} for r in rows if r['state'] == 'complete']
    baseline = reference['summary']
    flat.append({'method': 'inverse_ising', 'arm': None, 'residual': None,
                 'gauge_augmentation': None, 'seed': None,
                 'train_hamiltonians': None, 'shots': plan['data']['shots'],
                 'mse': baseline['metrics']['mse'],
                 'D5_reference_sign': baseline['metrics']['reference_sign_D5']['two_channel_full'],
                 'gauge_prediction_mse': None, 'elapsed_seconds': baseline['elapsed_seconds'],
                 'stop_reason': 'completed prior reference',
                 'provenance': 'immutable prior fit reused once; no new inverse fit'})
    _csv(run / 'comparison.csv', flat, tuple(flat[0]))
    contrasts = factorial_contrasts(rows)
    _json(run / 'factorial-contrasts.json', contrasts)
    text = '# Structural residual/gauge development study\n\n'
    text += f"CNNs complete: {status['completed']}/{status['total']}. Original-view validation selects checkpoints; gauge views are diagnostic only.\n\n"
    text += '| Method | Arm | Seed | State | Validation MSE |\n|---|---|---:|---|---:|\n'
    for row in rows:
        mse = row.get('metrics', {}).get('mse')
        text += f"| {row['method']} | {row['arm']} | {row['seed']} | {row['state']} | {mse if mse is not None else '—'} |\n"
    text += '\nData are read-only references to the frozen scaling cache. One matched inverse reference is reused without refitting. Reused development H and correlated records are not final confirmation.\n\n'
    text += 'Files: plan/source/data bindings; models/<task> checkpoints, histories, unaugmented predictions, per-H metrics and gauge diagnostics; status.json; comparison.csv; factorial-contrasts.json.\n'
    (run / 'RESULTS.md').write_text(text)
    return status


def progress(repo_root, config):
    """Read-only monitoring; no original cache or model output is rewritten."""
    root = Path(repo_root).resolve()
    config = validate_config(config)
    run = _safe_run(root, root / config['output_root'])
    if not (run / 'plan.json').exists():
        return {'state': 'not_prepared', 'completed': 0, 'total': len(_tasks(config)),
                'tasks': [], 'train_hamiltonians': config['selection']['train_hamiltonians'],
                'shots': config['selection']['shots'], 'inverse_reference': {'state': 'not_verified'}}
    plan = load_plan(root, run)
    if plan['config'] != config:
        raise ValueError('Config differs from the frozen structural run')
    return _report(run, plan, persist=False)


def run_campaign(repo_root, config, *, progress_callback=None):
    """Sequential CUDA queue; Ctrl+C saves after a bounded update/diagnostic chunk."""
    root = Path(repo_root).resolve()
    config = validate_config(config)
    requested = {'stop': False}
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda *_: requested.update(stop=True))
    stop = lambda: requested['stop']
    try:
        run = prepare(root, config, should_stop=stop, progress=progress_callback)
        if stop() or not (run / 'plan.json').exists():
            return progress(root, config)
        with pilot._io.campaign_lock(run):
            plan = load_plan(root, run)
            _, cache = _data_reference(root, config, expected=plan['data_reference'], load=True)
            for task in plan['tasks']:
                if stop():
                    break
                job = pilot._job(run, task)
                if _done(job, task, plan):
                    continue
                _run_neural(run, plan, cache, task, stop, progress_callback)
                _report(run, plan)
            return _report(run, plan)
    finally:
        signal.signal(signal.SIGINT, previous)
