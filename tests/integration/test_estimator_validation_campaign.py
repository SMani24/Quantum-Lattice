"""Synthetic end-to-end campaign controls, including guarded test exclusion."""
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from quantum_lattice.data import estimator_validation as campaign
from quantum_lattice.data.generation import _bank
from quantum_lattice.lattice import square_lattice
from quantum_lattice.observables.solvable import local_estimators


class ValidationCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        template = Path(__file__).resolve().parents[2]/"configs/conditioned_estimator_validation.json"
        self.config = json.loads(template.read_text())
        self.config.update(lattice_sizes=[3], bank_root="banks", pilot_selection="selection.json", derivation_pdf="proof.pdf")
        self.config["output_root"] = "codex material/artifacts/estimator_validation/synthetic"
        self.config["iid"].update(fields_per_size=1, replicates=3, records=12)
        self.config["stationary"].update(fields_per_size=1, replicas=2, total_sweeps=30)
        self.config["multistart"].update(random_fields_per_size=1, stress_fields_per_size=1,
            replicas=2, burn_in_sweeps=10, total_sweeps=80, thinning_sweeps=2,
            chunk_sweeps=10, backend="python", equal_budget_windows=[[10,30],[30,50],[60,80]],
            cumulative_ends=[30,50,80])
        self.config["statistics"].update(block_sizes=[2,3], probe_count=1)
        bc = {"lattice": {"L": 3, "type": "square_periodic"},
              "fields": {"channel":"z_only", "distribution":"uniform", "low":-.1,"high":.1},
              "dataset":{"hamiltonians":8,"seed":13,"split_fractions":{"train":.75,"validation":.125,"test":.125}}}
        directory = self.root/"banks/L3/run"
        (directory/"instances").mkdir(parents=True)
        self.instances = _bank(bc)
        lattice = square_lattice(3)
        verified = []
        for item in self.instances:
            if item["split"] == "test":
                # The test payload deliberately does not exist.
                verified.append({"id":item["id"],"split":"test","sha256":"not-a-real-test-hash"})
                continue
            fields = np.random.default_rng(np.random.SeedSequence(item["field_seed"])).uniform(-.1,.1,18)
            theta = np.random.default_rng(item["index"]).choice([-1,1],size=(12,9)).astype(np.int8)
            ends = lattice.edge_vertex_indices
            states = theta[:,ends[:,0]]*theta[:,ends[:,1]]
            values = local_estimators(lattice, fields, states)
            file = directory/"instances"/(item["id"]+".npz")
            np.savez(file,b_z=fields,states=states,star_mean=values["star"].mean(axis=0),
                     pair_mean=values["pair"].mean(axis=0),metadata_json=np.array(json.dumps({"instance":item})))
            verified.append({"id":item["id"],"split":item["split"],"sha256":campaign.sha(file)})
        (directory/"manifest.json").write_text(json.dumps({"instances":self.instances}))
        (directory/"config.json").write_text(json.dumps(bc))
        (directory/"verification.json").write_text(json.dumps({"passed":True,"instances":verified}))
        stress = next(item for item in self.instances if item["split"] != "test")
        (self.root/"selection.json").write_text(json.dumps([{"L":3,"id":stress["id"],"selection":"lowest saved scalar ESS"}]))
        (self.root/"proof.pdf").write_bytes(b"%PDF-synthetic-placeholder")

    def prepare(self):
        with patch.object(campaign, "_sources", return_value={}):
            return campaign.prepare(self.root, self.config)

    def test_declaration_guards_metadata_hash_and_test_payload(self):
        run = self.prepare()
        plan = campaign.load_plan(self.root, run)
        self.assertEqual(len(plan["tasks"]),14)
        self.assertTrue(all(t["id"] not in {i["id"] for i in self.instances if i["split"] == "test"} for t in plan["tasks"]))
        task = copy.deepcopy(plan["tasks"][0])
        task["id"] = next(i["id"] for i in self.instances if i["split"] == "test")
        with patch.object(np, "load", side_effect=AssertionError("no payload may be opened")):
            with self.assertRaisesRegex(ValueError,"non-test"):
                campaign._input(self.root,plan,task)
        changed = copy.deepcopy(plan)
        changed["tasks"][0]["id"] = task["id"]
        campaign.atomic_json(run/"plan.json", changed)
        with self.assertRaisesRegex(ValueError,"plan hash"):
            self.prepare()

    def test_two_worker_campaign_pause_resume_and_integrity(self):
        run = self.prepare()
        paused = campaign.run_campaign(self.root, run, workers=2, stage="multistart", max_steps=1)
        self.assertEqual(paused["state"],"paused")
        checkpoints = list((run/"jobs/multistart").glob("*/progress.json"))
        self.assertTrue(checkpoints)
        self.assertTrue(all(json.loads(p.read_text())["completed_sweeps"] <= 10 for p in checkpoints))
        finished = campaign.run_campaign(self.root,run,workers=2)
        self.assertEqual(finished["state"],"complete")
        self.assertEqual(finished["completed"],14)
        self.assertTrue((run/"RESULTS.md").is_file())
        rhats=json.loads((run/"multistart-rhat.json").read_text())["rows"]
        self.assertEqual(len(rhats),48)
        marker=next((run/"jobs/saved").glob("*/done.json"))
        before=marker.stat().st_mtime_ns
        self.assertEqual(campaign.run_campaign(self.root,run,workers=1)["state"],"complete")
        self.assertEqual(marker.stat().st_mtime_ns,before)
        coordinate=marker.parent/"coordinates.npz"
        coordinate.write_bytes(coordinate.read_bytes()+b"changed")
        with self.assertRaisesRegex(ValueError,"completed result changed"):
            campaign.run_campaign(self.root,run,workers=1)

    def test_bad_config_and_changed_bank_fail_before_work(self):
        for key, value in (("replicas",1),("thinning_sweeps",0),("total_sweeps",11)):
            bad=copy.deepcopy(self.config)
            bad["multistart"][key]=value
            with self.assertRaises(ValueError):
                campaign.validate_config(bad)
        run=self.prepare()
        manifest=self.root/"banks/L3/run/manifest.json"
        manifest.write_text(manifest.read_text()+" ")
        with self.assertRaisesRegex(ValueError,"pinned input/source changed"):
            campaign.load_plan(self.root,run)

    def test_low_disk_pauses_and_orphan_task_lease_blocks_duplicate(self):
        run=self.prepare()
        with patch.object(campaign.shutil, "disk_usage") as usage:
            usage.return_value.free=0
            self.assertEqual(campaign.run_campaign(self.root,run,workers=1)["state"],"paused")
        self.assertIn("disk",json.loads((run/"status.json").read_text())["pause_reason"])
        plan=campaign.load_plan(self.root,run)
        task=plan["tasks"][0]
        directory=run/"jobs"/task["stage"]/task["key"]
        directory.mkdir(parents=True)
        with (directory/".task.lock").open("a+") as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with patch.object(campaign,"_execute_task",side_effect=AssertionError("must not dispatch")):
                with self.assertRaisesRegex(RuntimeError,"task still active"):
                    campaign.execute_task(self.root,run,plan,task)

    def test_real_sigint_drains_bounded_chunks_and_exits_paused(self):
        self.config["multistart"]["total_sweeps"]=5000
        run=self.prepare()
        program=("from quantum_lattice.data.estimator_validation import run_campaign\n"
                 "if __name__ == '__main__':\n"
                 f"    run_campaign({str(self.root)!r},{str(run)!r},workers=2,stage='multistart')\n")
        env={**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1"}
        child=subprocess.Popen([sys.executable,"-u","-c",program],stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,text=True,env=env)
        try:
            for line in child.stdout:
                if "checkpoint |" in line:
                    child.send_signal(signal.SIGINT)
                    break
            output,_=child.communicate(timeout=20)
            self.assertEqual(child.returncode,0,output)
            self.assertIn("Pausing",output)
            self.assertEqual(json.loads((run/"status.json").read_text())["state"],"paused")
            for p in (run/"jobs/multistart").glob("*/progress.json"):
                self.assertLess(json.loads(p.read_text())["completed_sweeps"],5000)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()


if __name__ == "__main__":
    unittest.main()
