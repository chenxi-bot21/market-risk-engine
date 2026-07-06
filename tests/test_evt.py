import unittest

import numpy as np
from scipy.stats import genpareto, t as student_t

from marketrisk.evt import GPDTail, evt_var_es, fit_gpd_tail
from marketrisk.var import historical_var, parametric_var


def _t4_returns(n=5000, seed=9):
    rng = np.random.default_rng(seed)
    return student_t.rvs(df=4, size=n, random_state=rng) * 0.01


class FitTests(unittest.TestCase):
    def test_detects_heavy_tail_on_student_t(self):
        # t(4) has true tail index xi = 1/4; the fit should land in the
        # heavy-tail region, clearly above the thin-tail xi ~ 0.
        tail = fit_gpd_tail(_t4_returns(), threshold_q=0.95)
        self.assertGreater(tail.xi, 0.05)
        self.assertLess(tail.xi, 0.6)
        self.assertGreater(tail.beta, 0.0)
        self.assertEqual(tail.n_obs, 5000)
        self.assertAlmostEqual(tail.n_exceed / tail.n_obs, 0.05, delta=0.01)

    def test_near_zero_xi_on_gaussian(self):
        rng = np.random.default_rng(1)
        tail = fit_gpd_tail(rng.normal(0, 0.01, 5000))
        self.assertLess(abs(tail.xi), 0.35)          # thin tail ⇒ small xi
        self.assertTrue(np.isfinite(tail.var(0.995)))

    def test_validation(self):
        with self.assertRaises(ValueError):
            fit_gpd_tail(np.zeros(100))               # too short
        with self.assertRaises(ValueError):
            fit_gpd_tail(_t4_returns(), threshold_q=0.999)  # too few exceedances


class QuantileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = _t4_returns()
        cls.tail = fit_gpd_tail(cls.r)

    def test_es_exceeds_var_and_monotone_in_alpha(self):
        v995, e995 = self.tail.var(0.995), self.tail.es(0.995)
        self.assertGreater(e995, v995)
        self.assertGreater(self.tail.var(0.999), self.tail.var(0.99))

    def test_evt_beats_normal_in_the_far_tail(self):
        # On t(4) data the normal grossly understates the 99.9% tail;
        # EVT must sit above it (that's the point of EVT).
        mu, sd = self.r.mean(), self.r.std(ddof=1)
        self.assertGreater(self.tail.var(0.999), parametric_var(mu, sd, 0.999))

    def test_evt_consistent_with_historical_inside_sample(self):
        # At 99% (well inside the sample) EVT and historical should agree
        # within a reasonable band.
        h = historical_var(self.r, 0.99)
        self.assertAlmostEqual(self.tail.var(0.99), h, delta=0.35 * h)

    def test_alpha_below_threshold_rejected(self):
        with self.assertRaises(ValueError):
            self.tail.var(0.90)

    def test_es_requires_xi_below_one(self):
        bad = GPDTail(threshold=0.02, xi=1.2, beta=0.01,
                      n_obs=1000, n_exceed=50, threshold_q=0.95)
        with self.assertRaises(ValueError):
            bad.es(0.995)


class ConvenienceTests(unittest.TestCase):
    def test_one_call_wrapper(self):
        v, e = evt_var_es(_t4_returns(), alpha=0.995)
        self.assertGreater(e, v)
        self.assertGreater(v, 0)


if __name__ == "__main__":
    unittest.main()
