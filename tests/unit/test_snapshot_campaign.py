"""Synthetic scheduling, matched-record, exact-resume and corruption controls."""
import copy
import hashlib
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from quantum_lattice.data.state_storage import encode_states
from quantum_lattice.lattice import square_lattice
from quantum_lattice.models import snapshot_campaign as campaign
from quantum_lattice.models.snapshot_terminal import print_status

REPO=Path(__file__).resolve().parents[2]


def fixture(root):
    for name in ('src','experiments','configs'):(root/name).symlink_to(REPO/name,target_is_directory=True)
    bank=root/'codex material/bank';(bank/'instances').mkdir(parents=True)
    feature=root/'codex material/artifacts/fixture-features/L4';feature.mkdir(parents=True)
    rng=np.random.default_rng(799);lattice=square_lattice(4);instances=[];origins={};selected={'train':[],'validation':[]}
    for i,split in enumerate(['train']*3+['validation']*2):
        name=f'h{i}';instance={'id':name,'split':split,'hamiltonian_uid':f'uid-{i}','field_seed':[i]}
        tau=rng.choice(np.array([-1,1],np.int8),(8,16));ends=lattice.edge_vertex_indices
        states=tau[:,ends[:,0]]*tau[:,ends[:,1]];fields=rng.uniform(-.7,.7,32)
        metadata={'instance':instance,'retained_records':8,'chain':{'fields':fields.tolist()}}
        p=bank/'instances'/f'{name}.npz';np.savez(p,**encode_states(states,'packed_bits'),b_z=fields,edge_vertex_indices=ends,metadata_json=np.asarray(json.dumps(metadata)))
        p.with_suffix('.json').write_text(json.dumps({'id':name,'split':split,'records':8}))
        selected[split].append(name);instances.append(instance);origins[name]={'instance':instance,'input_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    instances.append({'id':'test-no-payload','split':'test','hamiltonian_uid':'unopened'})
    (bank/'manifest.json').write_text(json.dumps({'instances':instances}))
    (feature/'manifest.json').write_text(json.dumps({'L':4,'original_bank':str(bank.relative_to(root)),'selected_ids':selected,'origins':origins}))
    c=json.loads((REPO/'configs/snapshot_learning.json').read_text());c['feature_cache']=str(feature.parent.relative_to(root))
    c['output_root']='codex material/artifacts/snapshot_learning/campaigns/synthetic';c['selection']={'selection_seed':813,'train_hamiltonians':3,'validation_hamiltonians':2,'shot_counts':[2,4]};c['seeds']=[617]
    c['training'].update(batch_size_hamiltonians=2,max_epochs=2,validation_batch_size=1,shot_chunk_size=2)
    c['training']['early_stopping'].update(min_epochs=1,patience_epochs=1)
    c['inverse_ising'].update(max_iterations=3)
    c['execution'].update(checkpoint_updates=1,inverse_workers=1)
    return c


def run_cpu(root,c,**kwargs):
    # Public config remains CUDA-only; only this synthetic test substitutes the device resolver.
    with patch('quantum_lattice.models.snapshot_training.resolve_device',return_value=torch.device('cpu')),patch.object(campaign,'_gpu_binding',return_value={'accelerator':{'type':'cpu-synthetic'}}):
        return campaign.run_campaign(root,c,**kwargs)


class SnapshotCampaignTests(unittest.TestCase):
    def test_prepare_freezes_zero_updates_and_status_needs_no_GPU(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=fixture(root)
            self.assertEqual(campaign.progress(root,c)['state'],'not_prepared')
            run=campaign.prepare(root,c);plan=campaign.load_plan(root,run)
            self.assertEqual(len(plan['tasks']),6)
            self.assertFalse((run/'device-binding.json').exists());self.assertFalse((run/'models').exists())
            with patch.object(campaign,'_json',side_effect=AssertionError('Status must be read-only')):
                self.assertEqual(campaign.progress(root,c)['completed'],0)
            original=campaign._sha(run/'plan.json');campaign.prepare(root,c);self.assertEqual(original,campaign._sha(run/'plan.json'))
            cache=campaign.load_snapshot_cache(run/'cache')
            np.testing.assert_array_equal(cache.train.means(2),cache.train.spin_maps(2).mean(axis=1,dtype=np.float32))
            self.assertFalse((root/'codex material/bank/instances/test-no-payload.npz').exists())
            out=io.StringIO();print_status(campaign.progress(root,c),stream=out);self.assertIn('0/6',out.getvalue())

    def test_all_methods_complete_export_same_records_and_reject_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=fixture(root);status=run_cpu(root,c)
            self.assertEqual(status['completed'],6);run=root/c['output_root']
            seen={}
            for task in campaign.load_plan(root,run)['tasks']:
                job=run/'models'/task['key'];evaluation=json.loads((job/'evaluation.json').read_text())
                self.assertFalse(evaluation['test_payload_loaded']);self.assertFalse(evaluation['fields_used_as_inputs'])
                with np.load(job/'predictions.npz',allow_pickle=False) as z:
                    seen.setdefault(task['shots'],z['record_indices'].copy());np.testing.assert_array_equal(seen[task['shots']],z['record_indices'])
                    self.assertEqual(z['prediction'].shape,(2,32));self.assertTrue(np.isfinite(z['prediction']).all())
                if task['method']=='inverse_ising':self.assertEqual(evaluation['total_instances'],2)
            hashes={str(p.relative_to(run)):campaign._sha(p) for p in (run/'models').rglob('*') if p.is_file()}
            run_cpu(root,c)
            self.assertEqual(hashes,{str(p.relative_to(run)):campaign._sha(p) for p in (run/'models').rglob('*') if p.is_file()})
            p=next((run/'models').glob('*/predictions.npz'));p.write_bytes(p.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):campaign.progress(root,c)

    def test_partial_neural_pause_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=fixture(root);once={'done':False}
            def stop(event):
                if event.get('stage')=='training' and not once['done']:once['done']=True;signal.raise_signal(signal.SIGINT)
            paused=run_cpu(root,c,progress_callback=stop)
            self.assertEqual(paused['completed'],0)
            run=root/c['output_root'];job=next((run/'models').iterdir());payload=torch.load(job/'latest.pt',weights_only=False)
            self.assertEqual(payload['session']['engine']['step'],1)
            resumed=run_cpu(root,c);self.assertEqual(resumed['completed'],6)
            other=copy.deepcopy(c);other['output_root']+='-uninterrupted';run_cpu(root,other)
            for task in campaign.load_plan(root,run)['tasks']:
                with np.load(run/'models'/task['key']/'predictions.npz') as a,np.load(root/other['output_root']/'models'/task['key']/'predictions.npz') as b:np.testing.assert_array_equal(a['prediction'],b['prediction'])

    def test_inverse_pause_keeps_completed_fits_and_parallel_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=fixture(root);once={'done':False}
            def stop(event):
                if event.get('stage')=='inverse' and not once['done']:once['done']=True;signal.raise_signal(signal.SIGINT)
            run_cpu(root,c,progress_callback=stop);run=root/c['output_root']
            part=next((run/'models').glob('*/instances/*.json'));digest=campaign._sha(part)
            finished=run_cpu(root,c,workers=2)
            self.assertEqual(finished['completed'],6);self.assertEqual(campaign._sha(part),digest)

    def test_changed_settings_and_cpu_research_config_are_rejected(self):
        c=json.loads((REPO/'configs/snapshot_learning.json').read_text())
        for mutate in [lambda x:x['training'].update(device='cpu'),lambda x:x['selection'].update(shot_counts=[64,64]),lambda x:x.update(L=True),lambda x:x['inverse_ising'].update(ridge=0)]:
            v=copy.deepcopy(c);mutate(v)
            with self.assertRaises(ValueError):campaign.validate_config(v)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=fixture(root);campaign.prepare(root,c);changed=copy.deepcopy(c);changed['seeds']=[999]
            with self.assertRaisesRegex(ValueError,'differs'):campaign.prepare(root,changed)
            mispoint=copy.deepcopy(c);mispoint['output_root']+='-wrong-lattice'
            manifest=root/c['feature_cache']/'L4/manifest.json';data=json.loads(manifest.read_text());data['L']=3;manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'L differs'):campaign.prepare(root,mispoint)
            bad=copy.deepcopy(c);bad['output_root']='outside'
            with self.assertRaises(ValueError):campaign.prepare(root,bad)


if __name__=='__main__':unittest.main()
