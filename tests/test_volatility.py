import unittest

import numpy as np
import pandas as pd

from marketrisk.volatility import ewma_volatility, fit_garch


def _garch_series(n=2000, omega=2e-6, alpha=0.08, beta=0.90, seed=11) -> pd.Series:
    """Simulate a true GARCH(1,1) so the fitter has something to recover."""
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    s2 = omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return pd.Series(r, index=pd.bdate_range("2018-01-01", periods=n))


class EwmaTests(unittest.TestCase):
    def test_positive_and_full_length(self):
        r = _garch_series(500)
        vol = ewma_volatility(r)
        self.assertEqual(len(vol), len(r))
        self.assertTrue((vol > 0).all())

    def test_vol_rises_after_shock(self):
        r = pd.Series(np.full(200, 0.001))
        r.iloc[100] = -0.08                      # one large loss
        vol = ewma_volatility(r)
        self.assertGreater(vol.iloc[101], vol.iloc[100] * 2)

    def test_too_short_raises(self):
        with self.assertRaises(ValueError):
            ewma_volatility(pd.Series(np.zeros(10)))


class GarchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = _garch_series()
        cls.fit = fit_garch(cls.r)

    def test_stationary_and_sane(self):
        f = self.fit
        self.assertGreater(f.omega, 0)
        self.assertGreater(f.alpha, 0)
        self.assertGreater(f.beta, 0)
        self.assertLess(f.persistence, 1.0)
        self.assertTrue(np.isfinite(f.loglik))
        self.assertTrue((f.cond_vol > 0).all())

    def test_recovers_high_persistence(self):
        # True persistence is 0.98; MLE should land in the right neighbourhood.
        self.assertGreater(self.fit.persistence, 0.90)

    def test_forecast_reverts_to_long_run(self):
        fc = self.fit.forecast(horizon=100)
        self.assertEqual(fc.shape, (100,))
        self.assertTrue(np.all(fc > 0))
        gap_start = abs(fc[0] - self.fit.long_run_vol)
        gap_end = abs(fc[-1] - self.fit.long_run_vol)
        self.assertLessEqual(gap_end, gap_start + 1e-15)

    def test_too_short_raises(self):
        with self.assertRaises(ValueError):
            fit_garch(pd.Series(np.zeros(100)))


if __name__ == "__main__":
    unittest.main()
