"""Terminal restoration and redirected-output behavior for long simulations."""

import io
import os
import unittest
from unittest.mock import patch

from quantum_lattice.simulation.terminal import RunConsole


class RunConsoleTests(unittest.TestCase):
    def test_verification_has_its_own_progress_without_sampling_counters(self):
        output = io.StringIO()
        with RunConsole(4, interactive=False, output_stream=output) as console:
            console.render({"phase": "verifying", "completed": 1, "total": 4,
                            "elapsed_seconds": 2, "message": "Checking saved samples"})
        text = output.getvalue()
        self.assertIn("[VERIFYING]", text)
        self.assertIn("checked 1/4", text)
        self.assertNotIn("attempts 0", text)

    def test_redirected_output_throttles_and_has_no_ansi(self):
        output = io.StringIO()
        seconds = [0.0]
        status = {"completed": 0, "total": 2, "attempts_done": 100,
                  "attempts_total": 1000, "samples_done": 2, "samples_total": 20,
                  "elapsed_seconds": 10}
        with RunConsole(2, input_stream=io.StringIO(), output_stream=output,
                        clock=lambda: seconds[0]) as console:
            self.assertFalse(console.interactive)
            self.assertIsNone(console.poll())
            console.render(status)
            seconds[0] = 1.0
            console.render(status)
            self.assertEqual(len(output.getvalue().splitlines()), 1)
            seconds[0] = 5.0
            console.render(status)
            self.assertEqual(len(output.getvalue().splitlines()), 2)
            self.assertNotIn("\x1b", output.getvalue())
            self.assertIn("ETA ~0:01:30", output.getvalue())

    def test_pause_and_completion_are_immediate(self):
        output = io.StringIO()
        with RunConsole(1, interactive=False, output_stream=output,
                        clock=lambda: 0.0) as console:
            console.render({})
            console.render({"paused": True, "message": "Checkpoint saved"})
            console.render({"completed": 1})
        self.assertEqual(len(output.getvalue().splitlines()), 3)
        self.assertIn("[PAUSED]", output.getvalue())
        self.assertIn("[COMPLETE]", output.getvalue())

    def test_eta_override_and_pausing_status(self):
        output = io.StringIO()
        with RunConsole(1, interactive=False, output_stream=output,
                        clock=lambda: 0.0) as console:
            console.render({"attempts_done": 50, "attempts_total": 100,
                            "elapsed_seconds": 100, "eta_seconds": 20})
            console.render({"pausing": True, "eta_seconds": None,
                            "message": "Waiting for worker checkpoints"})
        lines = output.getvalue().splitlines()
        self.assertIn("ETA ~0:00:20", lines[0])
        self.assertIn("[PAUSING]", lines[1])
        self.assertIn("ETA ~--", lines[1])
        self.assertNotIn("[PAUSED]", lines[1])

    @unittest.skipUnless(os.name == "posix", "termios requires POSIX")
    def test_keys_without_enter_and_terminal_restoration_after_exception(self):
        import pty
        import termios

        master, slave = pty.openpty()
        original = termios.tcgetattr(slave)
        output = io.StringIO()
        try:
            with os.fdopen(os.dup(slave), "r") as keyboard:
                with patch.object(output, "isatty", return_value=True):
                    with patch.dict(os.environ, {"TERM": "xterm"}):
                        with self.assertRaisesRegex(RuntimeError, "example"):
                            with RunConsole(2, input_stream=keyboard,
                                            output_stream=output) as console:
                                self.assertTrue(console.interactive)
                                self.assertFalse(termios.tcgetattr(slave)[3] & termios.ICANON)
                                os.write(master, b"pRq")
                                self.assertEqual(console.poll(), "pause")
                                self.assertEqual(console.poll(), "resume")
                                self.assertEqual(console.poll(), "quit")
                                self.assertIsNone(console.poll())
                                console.render({"attempts_done": 5, "attempts_total": 10})
                                raise RuntimeError("example")
            self.assertEqual(termios.tcgetattr(slave), original)
            self.assertIn("\x1b[?25h", output.getvalue())
        finally:
            os.close(master)
            os.close(slave)


if __name__ == "__main__":
    unittest.main()
