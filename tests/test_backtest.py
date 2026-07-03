import unittest

import numpy as np
import pandas as pd

from marketrisk.backtest import (backtest_report, basel_traffic_light,
                                 christoffersen, kupiec_pof,
                                 rolling_var_backtest)


def _returns(n=800, seed=5) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0003, 0.01, n),
                     index=pd.bdate_range("2019-01-01", periods=n))


class RollingBacktestTests(unittest.TestCase):
    def test_shapes_and_violation_definition(self):
        r = _returns()
        bt = rolling_var_backtest(r, window=250, alpha=0.99)
        self.assertEqual(len(bt), len(r) - 250)
        self.assertTrue((bt["var"] > 0).all())
        # violation flag must equal "return breached -VaR", row by row
        expected = bt["return"] < -bt["var"]
        self.assertTrue((bt["violation"] == expected).all())

    def test_parametric_method_and_bad_method(self):
        r = _returns()
        bt = rolling_var_backtest(r, window=250, method="parametric")
        self.assertTrue((bt["var"] > 0).all())
        with self.assertRaises(ValueError):
            rolling_var_backtest(r, window=250, method="magic")

    def test_too_short_raises(self):
        with self.assertRaises(ValueError):
            rolling_var_backtest(_returns(200), window=250)


class KupiecTests(unittest.TestCase):
    def test_correct_coverage_passes(self):
        v = np.zeros(1000, dtype=bool)
        v[np.arange(0, 1000, 100)] = True          # 10 violations / 1000 @ 99%
        t = kupiec_pof(v, alpha=0.99)
        self.assertGreater(t.p_value, 0.05)
        self.assertFalse(t.reject_95)

    def test_far_too_many_violations_rejected(self):
        v = np.zeros(1000, dtype=bool)
        v[:50] = True                               # 5% breaches on a 99% model
        t = kupiec_pof(v, alpha=0.99)
        self.assertLess(t.p_value, 0.01)
        self.assertTrue(t.reject_95)

    def test_zero_violations_no_nan(self):
        t = kupiec_pof(np.zeros(500, dtype=bool), alpha=0.99)
        self.assertTrue(np.isfinite(t.statistic))
        self.assertTrue(np.isfinite(t.p_value))


class ChristoffersenTests(unittest.TestCase):
    def test_clustered_violations_fail_independence(self):
        v = np.zeros(1000, dtype=bool)
        v[100:110] = True                           # 10 breaches in a row
        ind, cc = christoffersen(v, alpha=0.99)
        self.assertTrue(ind.reject_95)
        self.assertTrue(cc.reject_95)

    def test_scattered_violations_pass_independence(self):
        v = np.zeros(1000, dtype=bool)
        v[np.arange(50, 1000, 95)] = True           # spread out
        ind, _cc = christoffersen(v, alpha=0.99)
        self.assertFalse(ind.reject_95)


class TrafficLightTests(unittest.TestCase):
    def _zone(self, k):
        v = np.zeros(250, dtype=bool)
        v[:k] = True
        return basel_traffic_light(v)["zone"]

    def test_zones(self):
        self.assertEqual(self._zone(0), "green")
        self.assertEqual(self._zone(4), "green")
        self.assertEqual(self._zone(5), "yellow")
        self.assertEqual(self._zone(9), "yellow")
        self.assertEqual(self._zone(10), "red")


class ReportTests(unittest.TestCase):
    def test_full_battery_keys(self):
        out = backtest_report(_returns(), window=250, alpha=0.99)
        self.assertEqual(len(out["tests"]), 3)
        self.assertIn(out["traffic_light"]["zone"], {"green", "yellow", "red"})
        self.assertEqual(len(out["backtest"]), 800 - 250)
        self.assertIsInstance(out["observed_violations"], int)


if __name__ == "__main__":
    unittest.main()
