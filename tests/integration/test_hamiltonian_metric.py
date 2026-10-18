"""Signed-field input/output contract for the standalone metric runner."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("hamiltonian_metric_runner", ROOT/"experiments/hamiltonian_metric/run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class HamiltonianMetricRunnerTests(unittest.TestCase):
    def data(self):
        return {"schema_version":1,"L":3,"true_b_z":[0.]*18,
                "recovered_b_z":[0.]*18,"sign_source":"provided_signed_fields"}

    def test_exact_reconstruction_reports_zero_and_signed_scope(self):
        value=runner.evaluate(self.data())
        self.assertEqual(value["d5_full_hamiltonian"],0.)
        self.assertEqual(value["d5_traceless_hamiltonian"],0.)
        self.assertEqual(value["qubits"],18)
        self.assertEqual(value["figure_rescaling"],"none; historical plot convention is not recovered")
        self.assertIn("estimated original",value["recovered_field_role"])

    def test_reference_sign_label_does_not_silently_fix_or_clip_inputs(self):
        value=self.data()
        value.update(true_b_z=[.2]*18,recovered_b_z=[-.2]*18,sign_source="reference_sign_oracle")
        result=runner.evaluate(value)
        self.assertEqual(result["sign_source"],"reference_sign_oracle")
        self.assertGreater(result["d5_full_hamiltonian"],.1)
        self.assertAlmostEqual(result["field_errors"]["z"]["rmse"],.4)

    def test_invalid_scope_shape_and_unknown_keys_are_rejected(self):
        for key,value in (("sign_source","guessed"),("true_b_z",[0.]*17),("extra",1)):
            with self.subTest(key=key):
                data=self.data();data[key]=value
                with self.assertRaises(ValueError):runner.evaluate(data)
        data=self.data();del data["recovered_b_z"]
        with self.assertRaises(ValueError):runner.evaluate(data)

    def test_explicit_input_exports_signed_vectors_scores_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/"input.json";source.write_text(json.dumps(self.data()))
            output=root/"artifacts/result"
            with patch.object(runner,"ARTIFACT_ROOT",root/"artifacts"),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(["--input",str(source),"--output",str(output)]),0)
            self.assertEqual(json.loads((output/"input.json").read_text()),self.data())
            self.assertEqual(json.loads((output/"metrics.json").read_text())["d5_full_hamiltonian"],0.)
            provenance=json.loads((output/"provenance.json").read_text())
            self.assertFalse(provenance["demo"])
            self.assertFalse(provenance["training_or_monte_carlo_performed"])
            self.assertTrue(provenance["input_sha256"])
            self.assertTrue((output/"RESULTS.md").exists())

    def test_existing_outputs_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);existing=root/"result";existing.mkdir()
            sentinel=existing/"keep.txt";sentinel.write_text("preserve")
            with patch.object(runner,"ARTIFACT_ROOT",root),self.assertRaisesRegex(ValueError,"already exists"):
                runner.main(["--output",str(existing)])
            self.assertEqual(sentinel.read_text(),"preserve")
            self.assertEqual(list(existing.iterdir()),[sentinel])

    def test_output_cannot_escape_artifact_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/"outside"
            with patch.object(runner,"ARTIFACT_ROOT",root/"artifacts"),self.assertRaisesRegex(ValueError,"subdirectory"):
                runner.main(["--output",str(output)])
            self.assertFalse(output.exists())


if __name__=="__main__":unittest.main()
