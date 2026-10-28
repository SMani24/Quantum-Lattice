"""Same-H ordering, rank diagnostics and exact-resume wrapper behavior."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from quantum_lattice.data.final_confirmation import load_confirmation_plan
from quantum_lattice.data.final_diagnostics import run_diagnostics, trace_diagnostics
from quantum_lattice.observables.conditioned import paired_estimators
from quantum_lattice.data.state_storage import decode_states
from quantum_lattice.lattice import square_lattice
from tests.unit import test_final_confirmation as confirmation_fixture

REPO=Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TraceDiagnosticTests(unittest.TestCase):
    def test_iid_rank_folded_and_conventional_Rhat_with_correct_split(self):
        rng=np.random.default_rng(871)
        a=rng.normal(size=(4,2001))
        result=trace_diagnostics(a,block_size_records=40)
        half=1000
        split=np.concatenate((a[:,:half],a[:,-half:]))
        W=np.var(split,axis=1,ddof=1).mean()
        B=half*np.var(split.mean(axis=1),ddof=1)
        expected=np.sqrt(((half-1)/half*W+B/half)/W)
        self.assertAlmostEqual(result["conventional_split_Rhat"],expected,places=14)
        self.assertLess(result["rank_folded_Rhat"],1.01)
        self.assertGreater(result["bulk_ESS"],1000)
        self.assertGreater(result["tail_ESS"],1000)
        self.assertEqual(result["dropped_middle_records_per_replica"],1)
        self.assertIsNotNone(result["batch_mean_MCSE"])
        # Positive affine scaling preserves both rank/folded diagnostics and
        # scales actual-value MCSE rather than substituting rank ESS for it.
        scaled=trace_diagnostics(3*a+10,block_size_records=40)
        self.assertAlmostEqual(scaled["rank_folded_Rhat"],result["rank_folded_Rhat"],places=14)
        self.assertAlmostEqual(scaled["batch_mean_MCSE"],3*result["batch_mean_MCSE"],places=14)

    def test_mean_shift_scale_shift_and_constant_traces_remain_flagged(self):
        rng=np.random.default_rng(998)
        base=rng.normal(size=(4,1000))
        shifted=base+np.array([0,0,3,3])[:,None]
        result=trace_diagnostics(shifted,block_size_records=20)
        self.assertGreater(result["rank_folded_Rhat"],1.1)
        self.assertTrue(result["flagged"])
        # Folded Rhat detects chains with similar means but different scales.
        scale=base*np.array([.2,.2,5,5])[:,None]
        result=trace_diagnostics(scale,block_size_records=20)
        self.assertGreater(result["folded_rank_normalized_split_Rhat"],1.1)
        constant=trace_diagnostics(np.ones((4,100)),block_size_records=10)
        self.assertIsNone(constant["rank_folded_Rhat"])
        self.assertIsNone(constant["bulk_ESS"])
        self.assertTrue(constant["flagged"])
        split_constants=trace_diagnostics(np.repeat([[0.],[1.]],20,axis=1),block_size_records=4)
        self.assertIsNone(split_constants["rank_folded_Rhat"])
        self.assertTrue(split_constants["flagged"])

    def test_invalid_shape_and_nonfinite_values_refused(self):
        for a in [np.ones(10),np.ones((1,10)),np.ones((2,3)),np.full((2,10),np.nan)]:
            with self.assertRaises(ValueError): trace_diagnostics(a)


class FreshDiagnosticWrapperTests(unittest.TestCase):
    def setUp(self):
        self.fixture=confirmation_fixture.FreshConfirmationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.run=self.fixture.prepare(sizes=(3,),count=2)
        self.cohort=load_confirmation_plan(REPO,self.run)
        for item in self.cohort["instances"]:
            self.fixture.complete(self.run,item["id"])
        self.output=self.fixture.root/"diagnostics"

    def execute(self,**kwargs):
        settings=dict(repo_root=REPO,count_per_size=1,replicas=4,workers=1,probe_count=2)
        settings.update(kwargs)
        return run_diagnostics(self.run,self.output,**settings)

    def test_same_H_panel_main_chain_reuse_distinct_starts_and_chronology(self):
        result=self.execute()
        self.assertEqual(result["state"],"complete")
        self.assertEqual(result["selected_H"],1)
        self.assertEqual(result["additional_chains"],3)
        self.assertFalse(result["cohort_filtering"])
        plan=read(self.output/"plan.json")
        selected=min(self.cohort["instances"],key=lambda x:x["hamiltonian_uid"])
        self.assertEqual(plan["selected_instances"],[selected])
        fingerprints=[x["physical_state_sha256"] for x in plan["initializations"][selected["id"]]]
        self.assertEqual(len(set(fingerprints)),4)
        directory=self.output/"summaries"/selected["id"]
        with np.load(directory/"traces.npz",allow_pickle=False) as a:
            traces=a["traces"]
            names=a["quantity_names"].tolist()
            self.assertEqual(traces.shape[:2],(4,8))
        with np.load(self.run/"instances"/selected["id"]/"records.npz",allow_pickle=False) as a:
            records=decode_states(a)
            fields=a["fields"]
        np.testing.assert_allclose(traces[0,:,names.index("log_weight")],records@fields,rtol=0,atol=1e-14)
        summary=read(directory/"summary.json")
        paired=paired_estimators(square_lattice(3),fields,records)
        for channel in ("star","pair"):
            for i in summary["probes"][channel]:
                for view in ("f","g"):
                    name=f"{view}_{channel}_{i}"
                    np.testing.assert_array_equal(traces[0,:,names.index(name)],paired[channel][view][:,i])
        # Eight records are inadequate for ESS400; warnings must be retained.
        self.assertGreater(len(summary["flagged_quantities"]),0)
        self.assertEqual(result["flagged_H"],1)

    def test_pause_resume_matches_direct_and_complete_reuse_is_immutable(self):
        paused=self.execute(max_seconds=0)
        self.assertEqual(paused["state"],"paused")
        self.assertEqual(paused["completed_additional_chains"],0)
        calls=[]
        def stop_after_two():
            calls.append(1)
            return len(calls)>2
        self.execute(stop_requested=stop_after_two)
        self.assertTrue(list((self.output/"chains").glob("*/replica-*/progress.json")))
        done=self.execute()
        self.assertEqual(done["state"],"complete")
        other=self.fixture.root/"direct-diagnostics"
        direct=run_diagnostics(self.run,other,repo_root=REPO,count_per_size=1,replicas=4,workers=1,probe_count=2)
        self.assertEqual(direct["state"],"complete")
        selected=read(self.output/"plan.json")["selected_instances"][0]["id"]
        with np.load(self.output/"summaries"/selected/"traces.npz") as a, np.load(other/"summaries"/selected/"traces.npz") as b:
            np.testing.assert_array_equal(a["traces"],b["traces"])
        before={str(p.relative_to(self.output)):sha(p) for p in self.output.rglob("*")
                if p.is_file() and p.name not in {"status.json","RESULTS.json"}}
        self.execute(max_seconds=0)
        after={str(p.relative_to(self.output)):sha(p) for p in self.output.rglob("*")
               if p.is_file() and p.name not in {"status.json","RESULTS.json"}}
        self.assertEqual(before,after)
        self.assertEqual(read(self.output/"status.json")["state"],"complete")

    def test_panel_change_corruption_and_incomplete_cohort_refused(self):
        self.execute()
        with self.assertRaisesRegex(ValueError,"panel/source changed"):
            self.execute(probe_count=3)
        selected=read(self.output/"plan.json")["selected_instances"][0]["id"]
        path=self.output/"summaries"/selected/"traces.npz"
        with path.open("ab") as handle: handle.write(b"tampered")
        with self.assertRaisesRegex(ValueError,"summary checksum"):
            self.execute()
        incomplete=self.fixture.prepare(name="incomplete")
        with self.assertRaisesRegex(ValueError,"incomplete"):
            run_diagnostics(incomplete,self.fixture.root/"incomplete-diagnostic",repo_root=REPO,
                            count_per_size=1,replicas=4,workers=1)

    def test_parallel_replica_workers_complete_fixed_panel(self):
        result=self.execute(workers=2)
        self.assertEqual(result["state"],"complete")
        self.assertEqual(result["completed_additional_chains"],3)


if __name__=="__main__":
    unittest.main()
