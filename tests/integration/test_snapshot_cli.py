"""CPU menu and actual CUDA end-to-end scheduling on tiny synthetic records."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'unit'))
from test_snapshot_campaign import fixture
from quantum_lattice.models.snapshot_terminal import main
from quantum_lattice.models.snapshot_campaign import load_plan


class SnapshotCliTests(unittest.TestCase):
    def test_status_and_menu_worker_preferences_without_GPU(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c=fixture(root);path=root/'local.json';path.write_text(json.dumps(c));out=io.StringIO()
            with contextlib.redirect_stdout(out):self.assertEqual(main(['--config',str(path),'--status'],repo_root=root),0)
            self.assertIn('0/6',out.getvalue());self.assertFalse((root/c['output_root']/'plan.json').exists())
            with contextlib.redirect_stdout(io.StringIO()),patch('builtins.input',side_effect=['3','2','2','0']):self.assertEqual(main(['--config',str(path)],repo_root=root),0)
            self.assertEqual(json.loads((root/c['output_root']/'preferences.json').read_text())['inverse_workers'],2)

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required; never substitute CPU for research runner')
    def test_actual_CUDA_full_tiny_pilot_and_completed_skip(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c=fixture(root);path=root/'local.json';path.write_text(json.dumps(c))
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(main(['--config',str(path),'--run','--workers','1'],repo_root=root),0)
            run=root/c['output_root'];status=json.loads((run/'status.json').read_text());self.assertEqual(status['completed'],6)
            binding=json.loads((run/'device-binding.json').read_text());self.assertEqual(binding['accelerator']['type'],'cuda')
            hashes={str(p.relative_to(run)):p.read_bytes() for p in (run/'models').rglob('*') if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(main(['--config',str(path),'--run','--workers','2'],repo_root=root),0)
            self.assertEqual(hashes,{str(p.relative_to(run)):p.read_bytes() for p in (run/'models').rglob('*') if p.is_file()})
            self.assertEqual(len(load_plan(root,run)['tasks']),6)


if __name__=='__main__':unittest.main()
