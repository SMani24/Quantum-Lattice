"""The training panel reports optimizer progress without sampling counters."""

import io
import unittest

from quantum_lattice.models.mlp_terminal import TrainingConsole


class TrainingConsoleTests(unittest.TestCase):
    def test_plain_display_has_training_metrics_and_no_sampling_counters(self):
        output = io.StringIO()
        with TrainingConsole(100, interactive=False, output_stream=output) as console:
            console.render({"step": 20, "epoch": 2, "train_mse": 0.1,
                            "validation_mse": 0.2, "best_validation_mse": 0.15,
                            "best_step": 10, "elapsed_seconds": 12,
                            "message": "Saved\ncheckpoint\x1b[bad"})
        text = output.getvalue()
        self.assertIn("Updates 20/100; epoch 2", text)
        self.assertIn("train MSE 0.1; validation MSE 0.2", text)
        self.assertIn("best validation MSE 0.15 at update 10", text)
        self.assertIn("ETA ~0:00:48", text)
        self.assertEqual(len(text.splitlines()), 1)
        self.assertNotIn("\x1b", text)
        for word in ("Hamiltonians", "samples", "attempts"):
            self.assertNotIn(word, text)

    def test_throttling_and_state_changes(self):
        output = io.StringIO()
        seconds = [0.0]
        with TrainingConsole(10, interactive=False, output_stream=output,
                             clock=lambda: seconds[0]) as console:
            console.render({"step": 1})
            seconds[0] = 1
            console.render({"step": 2})
            self.assertEqual(len(output.getvalue().splitlines()), 1)
            console.render({"step": 2, "state": "paused"})
            console.render({"step": 2, "state": "training"})
            console.render({"step": 10, "state": "trained"})
            console.render({"step": 10, "state": "evaluating"})
            console.render({"step": 10, "state": "evaluated"})
            self.assertEqual(len(output.getvalue().splitlines()), 6)
            self.assertIn("[PAUSED]", output.getvalue())
            self.assertIn("[EVALUATING]", output.getvalue())
            self.assertIn("[EVALUATED]", output.getvalue())

    def test_unknown_metrics_and_eta_override(self):
        output = io.StringIO()
        with TrainingConsole(10, interactive=False, output_stream=output) as console:
            self.assertIsNone(console.poll())
            console.render({"step": 2, "elapsed_seconds": 30,
                            "eta_seconds": None, "best_validation_mse": float("inf")})
        self.assertIn("best validation MSE -- at update --", output.getvalue())
        self.assertIn("ETA ~--", output.getvalue())


if __name__ == "__main__":
    unittest.main()
