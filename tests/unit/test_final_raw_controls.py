"""Nested record multiplicity, bound checkpoints and resumable raw controls."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import csv
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from quantum_lattice.data.state_storage import encode_states
from quantum_lattice.lattice import square_lattice
from quantum_lattice.models import final_raw_controls as raw
from quantum_lattice.models import final_campaign as campaign
from quantum_lattice.models.inverse_ising import fit_inverse_ising
from quantum_lattice.models.snapshot_cnn import make_snapshot_model, snapshot_model_spec

REPO=Path(__file__).resolve().parents[2]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))


class RecordSelectionTests(unittest.TestCase):
    def test_positions_are_reproducible_without_replacement(self):
        positions=raw.record_indices("uid-a",320,256,118)
        np.testing.assert_array_equal(positions,raw.record_indices("uid-a",320,256,118))
        self.assertEqual(len(set(positions)),256)
        self.assertTrue(np.all((positions>=0)&(positions<320)))
        self.assertFalse(np.array_equal(positions,raw.record_indices("uid-b",320,256,118)))
        # The run takes smaller budgets from this single common maximum bag.
        repeated=np.ones((320,32),dtype=np.int8)
        self.assertEqual(len(repeated[positions[:64]]),64)
        self.assertEqual(len(np.unique(repeated[positions[:64]],axis=0)),1)

    def test_invalid_selection_types_and_ranges_are_refused(self):
        for args in [("",320,256,0),("a",True,1,0),("a",320,True,0),
                     ("a",320,1.5,0),("a",320,321,0),("a",320,0,0),
                     ("a",320,256,True),("a",320,256,-1)]:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):raw.record_indices(*args)


class RawCheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        torch.set_num_threads(1)
        self.seed=2026102101
        self.states=np.empty((2,5,32),dtype=np.int8)
        for h in range(2):
            for shot in range(5):
                self.states[h,shot]=np.where((np.arange(32)+h+shot)%3==0,-1,1)

    def checkpoint(self,method):
        directory=self.root/f"L4-H7452-M256-seed{self.seed}-{method}"
        directory.mkdir(exist_ok=True)
        torch.manual_seed(119)
        model=make_snapshot_model(method)
        task={"key":directory.name,"method":method,"seed":self.seed,"shots":256,"train_hamiltonians":7452}
        payload={"task":task,"model_spec":snapshot_model_spec(method),"plan_sha256":"b"*64,
                 "model_state_dict":model.state_dict()}
        path=directory/"best.pt"
        torch.save(payload,path)
        done={"hashes":{"best.pt":raw.sha(path)},"task":task,"plan_sha256":"b"*64}
        write(directory/"done.json",done)
        return path,payload,done,model

    @contextmanager
    def cpu_device(self):
        original_fork=torch.random.fork_rng
        @contextmanager
        def cpu_fork(**kwargs):
            with original_fork(devices=[]):yield
        with ExitStack() as stack:
            stack.enter_context(patch.object(raw,"resolve_device",return_value=torch.device("cpu")))
            stack.enter_context(patch("torch.cuda.current_device",return_value=0))
            stack.enter_context(patch("torch.cuda.synchronize"))
            stack.enter_context(patch("torch.random.fork_rng",side_effect=cpu_fork))
            yield

    def test_snapshot_channel_order_and_both_model_paths_on_CPU(self):
        for method in ("mean_cnn","set_cnn"):
            with self.subTest(method=method):
                path,payload,done,model=self.checkpoint(method)
                # Explicit indexing independently reconstructs channels from
                # canonical e=2*(y*L+x)+orientation ordering.
                expected=np.array([[[[[self.states[h,s,2*(y*4+x)+axis] for x in range(4)]
                         for y in range(4)] for axis in (0,1)] for s in range(5)] for h in range(2)],dtype=np.float32)
                inputs=expected.mean(axis=1) if method=="mean_cnn" else expected
                model.eval()
                with torch.no_grad(): oracle=model(torch.from_numpy(inputs)).numpy()
                with self.cpu_device():
                    actual=raw._predict(path,self.states,method,self.seed)
                np.testing.assert_allclose(actual,oracle,rtol=1e-6,atol=1e-7)
                self.assertEqual(actual.shape,(2,2,4,4))

    def test_checkpoint_hash_task_model_and_plan_bindings_are_verified(self):
        path,payload,done,model=self.checkpoint("mean_cnn")
        with path.open("ab") as handle:handle.write(b"tampered")
        with self.assertRaisesRegex(ValueError,"checkpoint differs"):
            raw._predict(path,self.states,"mean_cnn",self.seed)
        for change in (lambda p:p["task"].update(seed=42),
                       lambda p:p["model_spec"].update(parameter_count=1),
                       lambda p:p.update(plan_sha256="c"*64)):
            altered=deepcopy(payload)
            change(altered)
            torch.save(altered,path)
            write(path.parent/"done.json",{**done,"hashes":{"best.pt":raw.sha(path)}})
            with self.assertRaisesRegex(ValueError,"task/architecture"):
                raw._predict(path,self.states,"mean_cnn",self.seed)


class RawControlWorkflowTests(unittest.TestCase):
    def setUp(self):
        parent=REPO/"codex material/artifacts/final_benchmark"
        parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix="test-raw-controls-",dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.confirmation=self.root/"confirmation"
        self.run=self.root/"run"
        self.source=self.root/"raw-models"
        self.seed=2026102101
        self.config={"enabled":True,"source_root":str(self.source.relative_to(REPO)),"size":4,
            "records":[256,64],"record_seed":2026102882,"seeds":[self.seed],
            "methods":["mean_cnn","set_cnn"],"ridge":.001,"max_iterations":3,"tolerance":1e-7}
        self.plan={"config":{"raw_panel":self.config},"fixture":True}
        self.items=[{"id":f"H{i}","L":4,"hamiltonian_uid":f"uid-{i}"} for i in range(2)]
        self.cp={"instances":self.items,"role":"confirmation"}
        self.fields=np.linspace(-1.7,1.7,64).reshape(2,32)
        self.data={"fields":self.fields,"hamiltonian_ids":np.array([i["id"] for i in self.items]),
                   "hamiltonian_uids":np.array([i["hamiltonian_uid"] for i in self.items])}
        self.records={}
        lattice=square_lattice(4)
        theta=np.ones(16,dtype=np.int8);theta[0]=-1
        flipped=theta[lattice.edge_vertex_indices[:,0]]*theta[lattice.edge_vertex_indices[:,1]]
        for i,item in enumerate(self.items):
            records=np.ones((320,32),dtype=np.int8)
            if i:records[1::2]=flipped
            self.records[item["id"]]=records
            path=self.confirmation/"instances"/item["id"]/"records.npz"
            path.parent.mkdir(parents=True)
            np.savez_compressed(path,**encode_states(records,"packed_bits"))
        for method in self.config["methods"]:
            path=self.source/f"L4-H7452-M256-seed{self.seed}-{method}"/"best.pt"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"checkpoint stub; CPU prediction mocked in workflow fixture")

    @contextmanager
    def patch_data(self):
        with patch.object(raw,"load_confirmation_plan",return_value=self.cp), \
             patch.object(raw,"load_confirmation_features",side_effect=lambda *a:dict(self.data)):
            yield

    def run_controls(self,**kwargs):
        options=dict(stop_requested=lambda:False,remaining_seconds=lambda:10000.)
        options.update(kwargs)
        with self.patch_data():
            return raw.run_raw_controls(REPO,self.run,self.plan,self.confirmation,**options)

    def predictor(self,path,states,method,seed):
        self.assertEqual(states.ndim,3)
        self.assertEqual(states.shape[2],32)
        self.assertTrue(np.isin(states,[-1,1]).all())
        self.assertEqual(seed,self.seed)
        return np.full((len(states),2,4,4),-.1,dtype=np.float32)

    def test_nested_bags_keep_duplicate_configurations_and_exclude_true_fields(self):
        calls=[]
        def predict(path,states,method,seed):
            calls.append((states.shape[1],method,states.copy()))
            return self.predictor(path,states,method,seed)
        solver_calls=[]
        def solve(states,**kwargs):
            self.assertNotIn("fields",kwargs)
            self.assertNotIn("true_fields",kwargs)
            self.assertEqual(states.shape[1],32)
            solver_calls.append(states.copy())
            return fit_inverse_ising(states,**kwargs)
        with patch.object(raw,"_predict",side_effect=predict), patch.object(raw,"fit_inverse_ising",side_effect=solve):
            result=self.run_controls()
        self.assertEqual(result,{"state":"complete"})
        by_budget={M:next(states for n,method,states in calls if n==M) for M in (256,64)}
        np.testing.assert_array_equal(by_budget[64],by_budget[256][:,:64])
        self.assertEqual(len(np.unique(by_budget[256][0],axis=0)),1)
        self.assertEqual(len(by_budget[256][0]),256)
        selection=raw.read(self.run/"raw_controls/record-selection.json")
        for i,item in enumerate(self.items):
            positions=np.asarray(selection["indices"][i])
            np.testing.assert_array_equal(by_budget[256][i],self.records[item["id"]][positions])
        self.assertEqual([len(a) for a in solver_calls],[256,256,64,64])
        with np.load(self.run/f"raw_controls/M256-mean_cnn-seed{self.seed}/predictions.npz") as a:
            self.assertTrue(np.all(a["prediction"]<0))
        with patch.object(raw,"_predict",side_effect=AssertionError("completed network reran")), \
             patch.object(raw,"fit_inverse_ising",side_effect=AssertionError("completed solver reran")):
            self.assertEqual(self.run_controls(),{"state":"complete"})

    def test_inverse_iteration_pause_resume_matches_direct_solution(self):
        pause={"requested":False,"used":False}
        initials=[]
        def solve(states,**kwargs):
            initials.append(kwargs.get("initial"))
            original=kwargs["callback"]
            def checkpoint(value):
                if not pause["used"]:
                    pause.update(requested=True,used=True)
                return original(value)
            return fit_inverse_ising(states,**{**kwargs,"callback":checkpoint})
        with patch.object(raw,"_predict",side_effect=self.predictor), patch.object(raw,"fit_inverse_ising",side_effect=solve):
            first=self.run_controls(stop_requested=lambda:pause["requested"])
            self.assertEqual(first,{"state":"paused"})
            path=self.run/"raw_controls/M256-inverse/instances/H0.checkpoint.json"
            saved=raw._read_checked(path)
            self.assertEqual(saved["optimizer_state"]["iteration"],1)
            pause["requested"]=False
            self.assertEqual(self.run_controls(),{"state":"complete"})
        self.assertIsNone(initials[0])
        self.assertIsNotNone(initials[1])
        self.assertEqual(initials[1]["iteration"],1)
        resumed=self.run/"raw_controls/M256-inverse/evaluation/predictions.npz"
        with np.load(resumed) as a: recovered=a["native_signed_prediction"].copy()
        self.run=self.root/"direct"
        with patch.object(raw,"_predict",side_effect=self.predictor):
            self.assertEqual(self.run_controls(),{"state":"complete"})
        with np.load(self.run/"raw_controls/M256-inverse/evaluation/predictions.npz") as a:
            np.testing.assert_array_equal(recovered,a["native_signed_prediction"])

    def test_selection_and_completed_inverse_corruption_are_refused(self):
        with patch.object(raw,"_predict",side_effect=self.predictor): self.run_controls()
        selection_path=self.run/"raw_controls/record-selection.json"
        selection=raw.read(selection_path)
        changed=deepcopy(selection);changed["indices"][0][0]=-1
        write(selection_path,changed)
        with self.assertRaisesRegex(ValueError,"selection changed"):
            self.run_controls()
        write(selection_path,selection)
        path=self.run/"raw_controls/M256-inverse/instances/H0.json"
        value=raw.read(path);value["payload"]["result"]["fields"][0]+=.1
        write(path,value)
        with self.assertRaisesRegex(ValueError,"state checksum"):
            self.run_controls()

    def test_inverse_checkpoint_corruption_and_early_pause_are_explicit(self):
        self.assertEqual(self.run_controls(stop_requested=lambda:True),{"state":"paused"})
        self.assertEqual(self.run_controls(remaining_seconds=lambda:0),{"state":"paused"})
        pause={"requested":False}
        def solve(states,**kwargs):
            original=kwargs["callback"]
            def checkpoint(value):
                pause["requested"]=True
                return original(value)
            return fit_inverse_ising(states,**{**kwargs,"callback":checkpoint})
        with patch.object(raw,"_predict",side_effect=self.predictor),patch.object(raw,"fit_inverse_ising",side_effect=solve):
            self.assertEqual(self.run_controls(stop_requested=lambda:pause["requested"]),{"state":"paused"})
        path=self.run/"raw_controls/M256-inverse/instances/H0.checkpoint.json"
        value=raw.read(path);value["payload"]["optimizer_state"]["iteration"]+=1
        write(path,value)
        with self.assertRaisesRegex(ValueError,"state checksum"):
            self.run_controls()

    def test_raw_and_observable_identity_order_must_match(self):
        self.data["hamiltonian_uids"]=self.data["hamiltonian_uids"][::-1]
        with self.assertRaisesRegex(ValueError,"identity order"):
            self.run_controls()


class RawComparisonReportTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix="ql-raw-report-fixture-")
        self.addCleanup(temporary.cleanup)
        self.run=Path(temporary.name)
        self.seeds=[2026102101,2026102102,2026102103,2026102104,2026102105]
        self.plan={"config":{"raw_panel":{"records":[256,64],"seeds":self.seeds,
                    "methods":["mean_cnn","set_cnn"]},
                    "statistics":{"bootstrap_samples":100,"seed":99}}}

    def evaluation(self,path):
        path=Path(path)
        key=path.parent.name if path.name=="evaluation" else path.name
        match=re.fullmatch(r"M(256|64)-(inverse|mean_cnn|set_cnn)(?:-seed(\d+))?",key)
        self.assertIsNotNone(match)
        M,method,seed=match.groups()
        offset=0. if seed is None else .001*(int(seed)-self.seeds[0])
        base={"inverse":.1,"set_cnn":.2,"mean_cnn":.4}[method]
        mse=np.array([base,base+.1,base+.2])+offset+(.02 if M=="64" else 0)
        d5=np.array([base/2,base/2+.05,base/2+.1])+offset
        return {"hamiltonians":3,"metrics":{"mse":float(mse.mean()),
                    "reference_sign_D5":{"two_channel_full":float(d5.mean())}},
                "per_hamiltonian":[{"hamiltonian_uid":uid,"mse":float(mse[i]),
                    "reference_sign_D5_two_channel_full":float(d5[i])}
                    for i,uid in enumerate(["A","B","C"])]}

    def test_one_solver_control_per_budget_and_paired_H_secondary_exports(self):
        with patch.object(campaign,"load_evaluation",side_effect=self.evaluation) as loader:
            campaign._raw_comparisons(self.run,self.plan)
        self.assertEqual(loader.call_count,22)  # Twenty NN seed cells; two solver controls.
        inverse_calls=[call for call in loader.call_args_list if "-inverse" in str(call.args[0])]
        self.assertEqual(len(inverse_calls),2)
        report=raw.read(self.run/"raw_controls/paired-comparisons.json")
        self.assertEqual(len(report["contrasts"]),12)
        self.assertIn("secondary",report["scope"])
        for contrast in report["contrasts"]:
            self.assertIs(contrast["statistical_primary"],False)
            self.assertEqual(contrast["hamiltonians"],3)
            self.assertEqual(contrast["optimizer_seeds"],5)
            self.assertEqual(len(contrast["per_seed"]),5)
            self.assertIn("one inverse fit per H/budget",contrast["control_scope"])
            if contrast["comparison"]=="set_cnn_minus_mean_cnn":
                expected=-.2 if contrast["metric"]=="mse" else -.1
                self.assertAlmostEqual(contrast["difference_left_minus_right"],expected)
                np.testing.assert_allclose(contrast["bootstrap_95ci_difference"],[expected,expected],atol=1e-14)
            elif contrast["comparison"]=="mean_cnn_minus_inverse":
                self.assertGreater(contrast["difference_left_minus_right"],0)
        with (self.run/"raw_controls/model-scores.csv").open(newline="") as handle:
            rows=list(csv.DictReader(handle))
        self.assertEqual(len(rows),22)
        controls=[row for row in rows if row["method"]=="inverse"]
        self.assertEqual({int(row["records"]) for row in controls},{256,64})
        self.assertEqual(len(controls),2)
        self.assertTrue(all(row["seed"]=="not_applicable" for row in controls))

    def test_UID_misalignment_is_refused_before_report_export(self):
        def permuted(path):
            result=self.evaluation(path)
            if "mean_cnn" in str(path):
                result["per_hamiltonian"]=result["per_hamiltonian"][::-1]
            return result
        with patch.object(campaign,"load_evaluation",side_effect=permuted):
            with self.assertRaisesRegex(ValueError,"Hamiltonian order differs"):
                campaign._raw_comparisons(self.run,self.plan)
        self.assertFalse((self.run/"raw_controls/model-scores.csv").exists())


if __name__=="__main__":
    unittest.main()
