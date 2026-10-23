"""Synthetic checks for matched factorial queues, immutable data and diagnostics."""
import copy
import hashlib
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_snapshot_scaling_engine import fixture as scaling_fixture
from quantum_lattice.models import snapshot_campaign as pilot
from quantum_lattice.models import snapshot_scaling as scaling
from quantum_lattice.models import snapshot_structure_campaign as structure

REPO = Path(__file__).resolve().parents[2]


def fixture(root):
    source = scaling_fixture(root)
    scaling.prepare(root, source)
    config = json.loads((REPO / 'configs/snapshot_structure.json').read_text())
    config.update(source_campaign=source['output_root'],
                  output_root='codex material/artifacts/snapshot_structure/campaigns/synthetic')
    config['selection'] = {'selection_seed': 813, 'train_hamiltonians': 2,
                           'validation_hamiltonians': 2, 'shots': 4}
    config['seeds'] = [617]
    for name in ('training', 'inverse_ising', 'inverse_reference'):
        config[name] = copy.deepcopy(source[name])
    config['execution'] = {'checkpoint_updates': 1}
    config['diagnostics'] = {'gauge_masks': 2, 'gauge_seed': 791}
    return config


def binding(run):
    value = {'schema_version': 1, 'accelerator': {'type': 'cpu-synthetic'}}
    path = run / 'device-binding.json'
    if path.exists() and structure._read(path) != value:
        raise ValueError('Synthetic device binding changed')
    if not path.exists(): structure._json(path, value)
    return value


def run_cpu(root, config, **kwargs):
    with patch('quantum_lattice.models.snapshot_structure_training.resolve_device',
               return_value=torch.device('cpu')), \
            patch.object(pilot, '_gpu_binding', side_effect=binding), \
            patch.object(pilot, '_run_inverse', side_effect=AssertionError('No new inverse fits')):
        return structure.run_campaign(root, config, **kwargs)


def hashes(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob('*') if p.is_file()}


class StructureEngineTests(unittest.TestCase):
    def test_default_design_and_invalid_science(self):
        config = json.loads((REPO / 'configs/snapshot_structure.json').read_text())
        structure.validate_config(config)
        tasks = structure._tasks(config)
        self.assertEqual(len(tasks), 40)
        self.assertEqual(len({t['key'] for t in tasks}), 40)
        self.assertEqual({t['arm'] for t in tasks}, {'P0', 'PG', 'R0', 'RG'})
        self.assertEqual({t['seed'] for t in tasks}, set(range(2026102301, 2026102306)))
        for mutate in (lambda c: c['factors'].update(residual=[0, 1]),
                       lambda c: c['factors'].update(gauge_augmentation=[True]),
                       lambda c: c['selection'].update(train_hamiltonians=True),
                       lambda c: c['training'].update(device='cpu'),
                       lambda c: c.update(source_campaign='../escape'),
                       lambda c: c['diagnostics'].update(gauge_masks=0)):
            bad = copy.deepcopy(config); mutate(bad)
            with self.assertRaises(ValueError): structure.validate_config(bad)
        reduced = copy.deepcopy(config)
        reduced['factors']['gauge_augmentation'] = [False]
        self.assertEqual(len(structure._tasks(structure.validate_config(reduced))), 20)

    def test_prepare_uses_only_bound_prefix_and_does_not_reopen_original_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = fixture(root)
            source = root / config['source_campaign']
            before = hashes(source)
            bank = root / 'codex material/bank/instances'
            bank.rename(bank.with_name('originals-unavailable'))
            with patch.object(pilot, 'prepare_snapshot_cache', side_effect=AssertionError('No raw rereads')):
                run = structure.prepare(root, config)
            plan = structure.load_plan(root, run)
            self.assertEqual(plan['updates_per_epoch'], 1)
            self.assertEqual(plan['max_optimizer_updates'], 2)
            self.assertEqual(before, hashes(source))
            self.assertFalse((run / 'cache').exists())
            self.assertFalse((run / 'models').exists())
            self.assertFalse((run / 'device-binding.json').exists())
            _, data = structure._data_reference(root, config, load=True)
            old = pilot.load_snapshot_cache(source / 'cache')
            np.testing.assert_array_equal(data.train.states, old.train.states[:2])
            np.testing.assert_array_equal(data.validation.record_indices, old.validation.record_indices)
            self.assertFalse(data.train.states.flags.writeable)
            with patch.object(structure, '_json', side_effect=AssertionError('Read-only status')):
                self.assertEqual(structure.progress(root, config)['completed'], 0)
            self.assertFalse((root / 'codex material/bank/instances/test-no-payload.npz').exists())

    def test_all_arms_resume_with_identical_predictions_and_complete_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = fixture(root)
            paused = []
            def stop_at_augmented_residual(event):
                if event.get('stage') == 'training' and event['task'].endswith('-RG') and not paused:
                    paused.append(True); signal.raise_signal(signal.SIGINT)
            status = run_cpu(root, config, progress_callback=stop_at_augmented_residual)
            self.assertTrue(paused)
            self.assertEqual(status['completed'], 3)
            self.assertEqual(status['tasks'][3]['state'], 'paused')
            self.assertEqual(run_cpu(root, config)['completed'], 8)
            other = copy.deepcopy(config); other['output_root'] += '-whole'
            self.assertEqual(run_cpu(root, other)['completed'], 8)
            run = root / config['output_root']; plan = structure.load_plan(root, run)
            _, data = structure._data_reference(root, config, load=True)
            for task in plan['tasks']:
                job = run / 'models' / task['key']
                whole = root / other['output_root'] / 'models' / task['key']
                with np.load(job / 'predictions.npz') as pred, np.load(whole / 'predictions.npz') as ref:
                    np.testing.assert_array_equal(pred['prediction'], ref['prediction'])
                    np.testing.assert_array_equal(pred['record_indices'], data.validation.record_indices)
                    np.testing.assert_array_equal(pred['hamiltonian_uids'], data.validation.uids)
                    mse = float(np.mean((pred['prediction'].astype(float)-pred['target'].astype(float))**2))
                evaluation = structure._read(job / 'evaluation.json')
                self.assertAlmostEqual(mse, evaluation['metrics']['mse'], places=14)
                with np.load(job / 'gauge-diagnostics.npz') as diag:
                    self.assertEqual(diag['squared_prediction_difference'].shape, (2, 2))
                    self.assertEqual(diag['squared_prediction_difference'].mean(),
                                     evaluation['gauge_diagnostics']['gauge_prediction_mse'])
                    np.testing.assert_array_equal(diag['hamiltonian_uids'], data.validation.uids)
            self.assertEqual(len(structure._read(run / 'factorial-contrasts.json')), 2)
            self.assertEqual((run / 'comparison.csv').read_text().count('immutable prior fit reused once'), 1)
            before = hashes(run / 'models')
            run_cpu(root, config)
            self.assertEqual(before, hashes(run / 'models'))

    def test_pause_during_diagnostics_restarts_exports_without_more_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = fixture(root)
            def stop(event):
                if event.get('stage') == 'diagnostics': signal.raise_signal(signal.SIGINT)
            result = run_cpu(root, config, progress_callback=stop)
            self.assertEqual(result['completed'], 0)
            run = root / config['output_root']; task = structure._tasks(config)[0]
            path = run / 'models' / task['key'] / 'latest.pt'
            before = torch.load(path, map_location='cpu', weights_only=True)['session']['engine']['step']
            self.assertEqual(run_cpu(root, config)['completed'], 8)
            after = torch.load(path, map_location='cpu', weights_only=True)['session']['engine']['step']
            self.assertEqual(before, after)

    def test_corrupt_reference_diagnostic_or_factor_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = fixture(root)
            run = structure.prepare(root, config)
            mismatch = copy.deepcopy(config); mismatch['seeds'] = [618]
            with self.assertRaisesRegex(ValueError, 'differs'): structure.prepare(root, mismatch)
            bad = copy.deepcopy(config); bad['selection']['selection_seed'] += 1
            bad['output_root'] += '-mismatch'
            with self.assertRaisesRegex(ValueError, 'selection differs'): structure.prepare(root, bad)
            def pause(event):
                if event.get('stage') == 'training': signal.raise_signal(signal.SIGINT)
            run_cpu(root, config, progress_callback=pause)
            path = run / 'models' / structure._tasks(config)[0]['key'] / 'latest.pt'
            saved = path.read_bytes(); payload = torch.load(path, map_location='cpu', weights_only=True)
            for mutate in (lambda p: p['session'].update(gauge_augmentation=True),
                           lambda p: p['session']['train_uids'].reverse(),
                           lambda p: p['session']['augmentation_spec'].update(seed=999)):
                corrupted = copy.deepcopy(payload); mutate(corrupted); torch.save(corrupted, path)
                with self.assertRaisesRegex(ValueError, 'model binding'): structure.progress(root, config)
                path.write_bytes(saved)
            self.assertEqual(run_cpu(root, config)['completed'], 8)
            diag = next((run / 'models').rglob('gauge-diagnostics.npz'))
            original = diag.read_bytes(); diag.write_bytes(original+b'changed')
            with self.assertRaisesRegex(ValueError, 'diagnostic changed'): structure.progress(root, config)
            diag.write_bytes(original)
            source = root / config['source_campaign'] / 'cache/train.npz'
            original = source.read_bytes(); source.write_bytes(original+b'changed')
            with self.assertRaises(ValueError): structure.progress(root, config)
            source.write_bytes(original)

    def test_factorial_contrasts_do_not_invent_missing_cells_or_seeds(self):
        rows = [{'state':'complete','method':'mean_cnn','seed':1,'arm':arm,'metrics':{'mse':value}}
                for arm,value in [('P0',10.),('PG',7.),('R0',8.),('RG',4.)]]
        contrast = structure.factorial_contrasts(rows)[0]
        self.assertEqual(contrast['residual_main_effect_mse'], -2.5)
        self.assertEqual(contrast['gauge_main_effect_mse'], -3.5)
        self.assertEqual(contrast['interaction_mse'], -1.)
        self.assertEqual(structure.factorial_contrasts(rows[:-1]), [])


if __name__ == '__main__': unittest.main()
