"""Small controls for the separate, checkpointed sampling diagnostic."""
import copy
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from quantum_lattice.lattice import square_lattice
from quantum_lattice.simulation.replica_exchange import ReplicaExchangeChain
from quantum_lattice.simulation.solvable import StarMetropolisChain

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / 'experiments/l24_sampling_diagnostic/run.py'
from quantum_lattice.data import sampling_diagnostic as diag


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        # Copy only source text to permit real CLI/spawn tests with isolated ROOT.
        for name in ('experiments/l24_sampling_diagnostic/run.py',
                     'src/quantum_lattice/simulation/replica_exchange.py',
                     'src/quantum_lattice/data/sampling_diagnostic.py'):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((REPO / name).read_bytes())
        self.config = json.loads((REPO/'configs/l24_sampling_diagnostic.json').read_text())
        self.config.update(reference_campaign='codex material/artifacts/estimator_validation/reference',
                           output_root='codex material/artifacts/estimator_validation/diagnostic',
                           L=3, flagged_star=0, replicas=2, backend='python', total_sweeps=40,
                           burn_in_sweeps=4, thinning_sweeps=3, chunk_sweeps=7, block_records=2)
        self.config['tempering'] = {'levels': 4, 'swap_interval_sweeps': 2}
        self.config['execution'] = {'workers': 2}
        self.lattice = square_lattice(3)
        self.fields = np.linspace(-.5, .8, 18)
        self.campaign = self.root / self.config['reference_campaign']
        self.campaign.mkdir(parents=True)
        bank = self.root/'bank'
        bank.mkdir()
        inputs = {}
        for name in ('manifest.json', 'config.json', 'verification.json'):
            path = bank/name
            path.write_text('{}\n')
            inputs[str(path.relative_to(self.root))] = diag.sha(path)
        entry = {'instance': {'split': 'train', 'index': 2371, 'id': 'hamiltonian_02371'},
                 'sha256': 'test-only-field-identifier'}
        tasks = []
        for r in range(2):
            key = f'L3-hamiltonian_02371-r{r:02d}'
            directory = self.campaign/'jobs/multistart'/key
            directory.mkdir(parents=True)
            chain = StarMetropolisChain(self.lattice, self.fields, [100, r], backend='python')
            chain.advance(12)
            np.savez(directory/'fields.npz', b_z=self.fields)
            diag.atomic_json(directory/'field.json', {'input_sha256': entry['sha256']})
            diag.atomic_json(directory/'progress.json', {'complete': True, 'completed_sweeps': 12,
                'spec': {'total_sweeps': 12}, 'chain': chain.snapshot()})
            names = ('fields.npz', 'field.json', 'progress.json')
            diag.atomic_json(directory/'done.json', {'files': {n: diag.sha(directory/n) for n in names}})
            tasks.append({'stage': 'multistart', 'key': key, 'L': 3, 'id': 'hamiltonian_02371'})
        reference = {'sources': {}, 'banks': {'3': {'directory': 'bank', 'entries': {'hamiltonian_02371': entry}}},
                     'inputs': inputs, 'tasks': tasks}
        diag.atomic_json(self.campaign/'plan.json', reference)
        diag.atomic_json(self.campaign/'plan-sha256.json', {'sha256': diag.sha(self.campaign/'plan.json')})
        self.run = self.root/self.config['output_root']
        self.config_path = self.root/'config.json'
        diag.atomic_json(self.config_path, self.config)
        self.plan = diag.prepare(self.root, self.config, self.run, self.config_path)

    def tearDown(self):
        self.temp.cleanup()

    def finish(self, job):
        for _ in range(10):
            if diag.advance_job(self.root, self.run, self.plan, job):
                return
        self.fail('bounded tiny job did not complete')

    def direct_records(self, job):
        if job['method'] == 'local':
            snapshot = diag.read_json(self.root/job['initial_checkpoint'])['chain']
            chain = StarMetropolisChain.from_snapshot(self.lattice, self.fields, snapshot, backend='python')
        else:
            chain = ReplicaExchangeChain(self.lattice, self.fields, job['seed'],
                betas=np.linspace(0,1,4), backend='python', swap_interval_sweeps=2)
        chain.advance(4)
        expected = []
        for _ in range(12):
            chain.advance(3)
            expected.append(chain.spins.copy())
        return np.array(expected), chain.snapshot()

    def test_committed_states_and_final_checkpoint_match_uninterrupted_chains(self):
        for job in self.plan['jobs'][:2]:
            self.finish(job)
            directory = self.run/'jobs'/job['key']
            packed = np.load(directory/'states_packed.npy', allow_pickle=False)
            states = 1-2*np.unpackbits(packed, axis=1, count=18, bitorder='little').astype(np.int8)
            expected, snapshot = self.direct_records(job)
            np.testing.assert_array_equal(states, expected)
            self.assertEqual(diag._read_checkpoint(directory, self.plan, job)['chain'], snapshot)
            with np.load(directory/'coordinates.npz') as data:
                # Independent vertex-0 support on the periodic square.
                h = states[:,[0,1,4,13]] @ self.fields[[0,1,4,13]]
                np.testing.assert_allclose(data['flagged_f'], np.exp(-h), atol=1e-14)
                np.testing.assert_allclose(data['flagged_g'], 1/np.cosh(h), atol=1e-14)

    def test_uncommitted_record_tail_is_overwritten_after_failed_checkpoint(self):
        job = self.plan['jobs'][1]
        diag.advance_job(self.root, self.run, self.plan, job)
        directory = self.run/'jobs'/job['key']
        previous = (directory/'checkpoint.json').read_bytes()
        with patch.object(diag, '_save_checkpoint', side_effect=OSError('interrupted commit')):
            with self.assertRaisesRegex(OSError, 'interrupted commit'):
                diag.advance_job(self.root, self.run, self.plan, job)
        self.assertEqual(previous, (directory/'checkpoint.json').read_bytes())
        self.finish(job)
        expected, _ = self.direct_records(job)
        packed = np.load(directory/'states_packed.npy')
        states = 1-2*np.unpackbits(packed, axis=1, count=18, bitorder='little').astype(np.int8)
        np.testing.assert_array_equal(states, expected)

    def test_corrupt_committed_record_and_done_output_are_rejected(self):
        job = self.plan['jobs'][0]
        diag.advance_job(self.root, self.run, self.plan, job)
        directory = self.run/'jobs'/job['key']
        records = np.load(directory/'states_packed.npy', mmap_mode='r+')
        records[0,0] ^= 1
        records.flush()
        with self.assertRaisesRegex(ValueError, 'Committed records hash'):
            diag.advance_job(self.root, self.run, self.plan, job)
        other = self.plan['jobs'][1]
        self.finish(other)
        (self.run/'jobs'/other['key']/'summary.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Completed output hash'):
            diag.advance_job(self.root, self.run, self.plan, other)

    def test_plan_guards_changed_inputs_and_numerical_environment(self):
        snapshot = self.root/self.plan['jobs'][0]['initial_checkpoint']
        snapshot.write_text(snapshot.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'Input hash'):
            diag.load_plan(self.root, self.run)

    def test_no_test_payload_is_opened_or_accepted(self):
        reference = diag.read_json(self.campaign/'plan.json')
        reference['banks']['3']['entries']['hamiltonian_02371']['instance']['split'] = 'test'
        diag.atomic_json(self.campaign/'plan.json', reference)
        diag.atomic_json(self.campaign/'plan-sha256.json', {'sha256':diag.sha(self.campaign/'plan.json')})
        other = self.run.parent/'forbidden'
        with self.assertRaisesRegex(ValueError, 'non-test'):
            diag.prepare(self.root, self.config, other)
        self.assertFalse(other.exists())

    def test_cli_spawn_bounded_pause_resume_and_worker_override(self):
        runner = self.root/'experiments/l24_sampling_diagnostic/run.py'
        command = [sys.executable, str(runner), 'run', '--config', str(self.config_path)]
        first = subprocess.run(command+['--workers','1','--max-chunks','1'], capture_output=True, text=True, timeout=30)
        self.assertEqual(first.returncode, 0, first.stderr)
        status = diag.read_json(self.run/'status.json')
        self.assertEqual(status['committed_sweeps'], 7)
        self.assertEqual(status['completed_jobs'], 0)
        resumed = subprocess.run(command+['--workers','2'], capture_output=True, text=True, timeout=30)
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual(diag.read_json(self.run/'status.json')['state'], 'complete')
        for job in self.plan['jobs']:
            expected, snapshot = self.direct_records(job)
            self.assertEqual(diag._read_checkpoint(self.run/'jobs'/job['key'],self.plan,job)['chain'], snapshot)
        before = {str(p):diag.sha(p) for p in (self.run/'jobs').rglob('done.json')}
        again = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertEqual(before, {str(p):diag.sha(p) for p in (self.run/'jobs').rglob('done.json')})

    def test_new_run_sweep_override_can_resume_from_frozen_plan(self):
        other = self.run.parent/'override'
        command = [sys.executable, str(self.root/'experiments/l24_sampling_diagnostic/run.py'),
                   'run', '--config',str(self.config_path),'--run-dir',str(other),'--workers','1']
        first = subprocess.run(command+['--sweeps','43','--max-chunks','1'], capture_output=True,text=True,timeout=30)
        self.assertEqual(first.returncode, 0, first.stderr)
        second = subprocess.run(command, capture_output=True,text=True,timeout=30)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(diag.read_json(other/'status.json')['state'], 'complete')
        self.assertEqual(diag.read_json(other/'plan.json')['config']['total_sweeps'], 43)

    def test_real_sigint_drains_one_chunk_and_checkpoint_remains_resumable(self):
        config = copy.deepcopy(self.config)
        config.update(total_sweeps=20000, thinning_sweeps=4, chunk_sweeps=1000)
        config['output_root'] = 'codex material/artifacts/estimator_validation/signal'
        path = self.root/'signal-config.json'
        diag.atomic_json(path, config)
        command = [sys.executable, str(self.root/'experiments/l24_sampling_diagnostic/run.py'),
                   'run', '--config', str(path), '--workers', '1']
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        checkpoint = self.root/config['output_root']/'jobs/local-r00/checkpoint.json'
        deadline = time.monotonic()+15
        while not checkpoint.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        if not checkpoint.exists():
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
            self.fail(f'No checkpoint before signal: {stdout} {stderr}')
        os.kill(process.pid, signal.SIGINT)
        stdout, stderr = process.communicate(timeout=15)
        self.assertEqual(process.returncode, 0, stderr)
        run = self.root/config['output_root']
        status = diag.read_json(run/'status.json')
        self.assertEqual(status['state'], 'incomplete')
        self.assertGreater(status['committed_sweeps'], 0)
        continued = subprocess.run(command+['--max-chunks','1'], capture_output=True,text=True,timeout=20)
        self.assertEqual(continued.returncode, 0, continued.stderr)
        self.assertGreater(diag.read_json(run/'status.json')['committed_sweeps'], status['committed_sweeps'])


if __name__ == '__main__':
    unittest.main()
