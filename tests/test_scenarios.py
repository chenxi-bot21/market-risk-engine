import unittest

import numpy as np
import pandas as pd

from marketrisk.scenarios import (PRESET_SCENARIOS, apply_scenario,
                                  historical_worst_windows,
                                  run_preset_scenarios, stress_report)


class ApplyScenarioTests(unittest.TestCase):
    NAMES = ["EQUITY", "BOND"]

    def test_pnl_is_weight_linear(self):
        res = apply_scenario([0.6, 0.4], {"EQUITY": -0.30, "BOND": 0.05}, self.NAMES)
        self.assertAlmostEqual(res.asset_pnl["EQUITY"], 0.6 * -0.30, places=12)
        self.assertAlmostEqual(res.asset_pnl["BOND"], 0.4 * 0.05, places=12)
        self.assertAlmostEqual(res.total_pnl, 0.6 * -0.30 + 0.4 * 0.05, places=12)

    def test_missing_assets_skipped_and_reported(self):
        res = apply_scenario([1.0, 0.0], {"EQUITY": -0.1, "CRYPTO": -0.5}, self.NAMES)
        self.assertIn("CRYPTO", res.skipped)
        self.assertEqual(res.applied, ["EQUITY"])
        self.assertAlmostEqual(res.total_pnl, -0.1)

    def test_weight_shape_validated(self):
        with self.assertRaises(ValueError):
            apply_scenario([1.0], {"EQUITY": -0.1}, self.NAMES)

    def test_preset_library_runs_for_partial_books(self):
        results = run_preset_scenarios([1.0], ["EQUITY"])
        self.assertEqual(len(results), len(PRESET_SCENARIOS))
        for r in results:
            self.assertTrue(np.isfinite(r.total_pnl))


class WorstWindowTests(unittest.TestCase):
    def _df_with_crash(self, n=300, crash_at=150, horizon=5):
        idx = pd.bdate_range("2022-01-03", periods=n)
        r = np.full(n, 0.001)
        r[crash_at:crash_at + horizon] = -0.05          # engineered 5-day crash
        return pd.DataFrame({"A": r, "B": np.zeros(n)}, index=idx), idx

    def test_finds_the_engineered_crash(self):
        df, idx = self._df_with_crash()
        worst = historical_worst_windows(df, [1.0, 0.0], horizon=5, top=1)
        self.assertEqual(len(worst), 1)
        self.assertAlmostEqual(worst["cum_return"].iloc[0], -0.25, places=10)
        self.assertEqual(worst["start"].iloc[0], idx[150])
        self.assertEqual(worst["end"].iloc[0], idx[154])

    def test_windows_do_not_overlap(self):
        df, _ = self._df_with_crash()
        worst = historical_worst_windows(df, [1.0, 0.0], horizon=5, top=3)
        ends = sorted(worst["end"])
        for a, b in zip(ends, ends[1:]):
            self.assertGreaterEqual((b - a).days, 5)

    def test_too_short_raises(self):
        df = pd.DataFrame({"A": [0.0] * 4},
                          index=pd.bdate_range("2024-01-01", periods=4))
        with self.assertRaises(ValueError):
            historical_worst_windows(df, [1.0], horizon=5)


class StressReportTests(unittest.TestCase):
    def test_keys_and_worst_loss_negative_for_crashy_data(self):
        idx = pd.bdate_range("2022-01-03", periods=400)
        rng = np.random.default_rng(2)
        df = pd.DataFrame({"EQUITY": rng.normal(0, 0.012, 400),
                           "BOND": rng.normal(0, 0.004, 400)}, index=idx)
        out = stress_report(df, [0.5, 0.5])
        self.assertEqual(len(out["presets"]), len(PRESET_SCENARIOS))
        self.assertEqual(len(out["worst_windows"]), 3)
        self.assertLess(out["worst_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
