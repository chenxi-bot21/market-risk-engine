import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from marketrisk.cli import run


class CliTests(unittest.TestCase):
    def test_end_to_end_synthetic_run(self):
        with tempfile.TemporaryDirectory() as d:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = run(["run", "--output", d, "--seed", "42"])
            self.assertEqual(code, 0)
            report = Path(d) / "market_risk_report.md"
            self.assertTrue(report.exists())
            text = report.read_text(encoding="utf-8")
            for token in ("VaR / ES by method", "component VaR", "GARCH(1,1)",
                          "Kupiec", "traffic light"):
                self.assertIn(token, text)
            out = buf.getvalue()
            self.assertIn("Basel zone", out)

    def test_custom_weights_and_alpha(self):
        with tempfile.TemporaryDirectory() as d:
            with contextlib.redirect_stdout(io.StringIO()):
                code = run(["run", "--output", d, "--alpha", "0.975",
                            "--weights", "0.4,0.3,0.2,0.1"])
            self.assertEqual(code, 0)
            text = (Path(d) / "market_risk_report.md").read_text(encoding="utf-8")
            self.assertIn("**97%**", text)  # int(0.975*100)


if __name__ == "__main__":
    unittest.main()
