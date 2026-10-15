"""The training panel reports optimizer progress without sampling counters."""

import io
import unittest

from quantum_lattice.models.mlp_terminal import TrainingConsole, training_context, training_stop_requested


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

    def test_early_stop_reports_finished_below_ceiling(self):
        output = io.StringIO()
        with TrainingConsole(200000, interactive=False, output_stream=output) as console:
            console.render({"step": 12000, "state": "trained", "elapsed_seconds": 30,
                            "stop_reason": "early_stopping", "early_stopping_wait": 20,
                            "early_stopping": {"enabled": True, "patience": 20}})
        self.assertIn("ETA ~0:00:00", output.getvalue())
        self.assertIn("early stop checks 20/20", output.getvalue())
        self.assertIn("stop reason early_stopping", output.getvalue())

    def test_context_labels_are_nested_and_restored(self):
        output = io.StringIO()
        with training_context(size=8, queue_index=3, queue_total=14):
            with training_context(mode="all_edges"):
                with TrainingConsole(10, interactive=False, output_stream=output) as console:
                    console.render({"step": 1})
            self.assertFalse(training_stop_requested())
        self.assertIn("L=8 | all_edges | model 3/14", output.getvalue())
        with TrainingConsole(10, interactive=False, output_stream=output) as console:
            self.assertEqual(console.training_context, {})

    def test_pending_quit_at_final_chunk_stops_parent_sequence(self):
        with training_context(size=4):
            with training_context(mode="paper_size"):
                with TrainingConsole(10, interactive=False, output_stream=io.StringIO()) as console:
                    # The key arrives after the engine's last loop poll.
                    console._commands.append("quit")
                    self.assertFalse(training_stop_requested())
            self.assertTrue(training_stop_requested())
        self.assertFalse(training_stop_requested())

    def test_quit_poll_sets_stop_flag_before_terminal_exit(self):
        with training_context():
            with TrainingConsole(10, interactive=False, output_stream=io.StringIO()) as console:
                console._commands.append("quit")
                self.assertEqual(console.poll(), "quit")
                self.assertTrue(training_stop_requested())

    def test_pending_pause_at_final_chunk_stops_parent_sequence(self):
        with training_context():
            with TrainingConsole(10, interactive=False, output_stream=io.StringIO()) as console:
                console._commands.append("pause")
            self.assertTrue(training_stop_requested())

    def test_pending_pause_then_resume_at_final_chunk_allows_next_model(self):
        with training_context():
            with TrainingConsole(10, interactive=False, output_stream=io.StringIO()) as console:
                console._commands.extend(("pause", "resume"))
            self.assertFalse(training_stop_requested())

    def test_interactive_panel_is_five_lines(self):
        output = io.StringIO()
        with TrainingConsole(10, interactive=False, output_stream=output) as console:
            console.interactive = True  # Exercise renderer without touching a terminal.
            console.render({"step": 1, "train_mse": .1, "validation_mse": .2})
            console.interactive = False
        self.assertEqual(output.getvalue().count("\n"), 5)


if __name__ == "__main__":
    unittest.main()
