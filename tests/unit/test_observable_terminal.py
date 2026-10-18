"""Menu controls never launch work implicitly and return after cooperative pause."""
import unittest

from quantum_lattice.models.observable_terminal import dashboard


class ObservableTerminalTests(unittest.TestCase):
    def drive(self, choices):
        output, launches = [], []
        iterator = iter(choices)
        def read(_):
            value = next(iterator)
            if isinstance(value, BaseException):
                raise value
            return value
        code = dashboard("/run", status=lambda: "0/40 complete",
                         launch=lambda **kw: launches.append(kw), read=read,
                         write=output.append)
        return code, launches, output

    def test_status_then_exit_never_starts(self):
        code, calls, output = self.drive(["4", "q"])
        self.assertEqual((code, calls), (0, []))
        self.assertIn("Results: /run/RESULTS.md", output)

    def test_size_selections_return_to_menu(self):
        code, calls, output = self.drive(["1", "2", "3", "q"])
        self.assertEqual(calls, [{"sizes": [4]}, {"sizes": [3]}, {"sizes": None}])
        self.assertEqual(code, 0)
        self.assertEqual(output.count("0/40 complete"), 4)

    def test_bad_choice_cannot_start(self):
        self.assertEqual(self.drive(["all", "q"])[1], [])

    def test_idle_ctrl_c_and_eof_exit(self):
        for error in (KeyboardInterrupt(), EOFError()):
            with self.subTest(error=type(error).__name__):
                self.assertEqual(self.drive([error])[:2], (0, []))


if __name__ == "__main__":
    unittest.main()
