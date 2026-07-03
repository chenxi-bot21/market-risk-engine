import unittest

import numpy as np
from scipy import stats

from marketrisk.var import (cornish_fisher_var, historical_es, historical_var,
                            monte_carlo_var_es, parametric_es, parametric_var,
                            var_summary)


class HistoricalTests(unittest.TestCase):
    def test_var_is_the_loss_quantile(self):
        # 100 returns: -0.10 once, rest small; 99% VaR ~ the worst-ish loss.
        r = np.full(100, 0.001)
        r[0] = -0.10
        v = historical_var(r, alpha=0.99)
        self.assertGreater(v, 0.0)
        self.assertLessEqual(v, 0.10 + 1e-12)

    def test_es_at_least_var(self):
        rng = np.random.default_rng(0)
        r = rng.standard_t(df=4, size=5000) * 0.01
        self.assertGreaterEqual(historical_es(r, 0.99), historical_var(r, 0.99))

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            historical_var(np.zeros(10), 0.99)          # too short
        with self.assertRaises(ValueError):
            historical_var(np.zeros(100), 0.3)          # bad alpha


class ParametricTests(unittest.TestCase):
    def test_matches_closed_form_standard_normal(self):
        # mu=0, sigma=1: 99% VaR = z_0.99 = 2.3263, ES = phi(z)/0.01 = 2.6652
        self.assertAlmostEqual(parametric_var(0.0, 1.0, 0.99),
                               stats.norm.ppf(0.99), places=10)
        self.assertAlmostEqual(parametric_es(0.0, 1.0, 0.99),
                               stats.norm.pdf(stats.norm.ppf(0.99)) / 0.01, places=10)

    def test_es_exceeds_var(self):
        self.assertGreater(parametric_es(0.0005, 0.012, 0.99),
                           parametric_var(0.0005, 0.012, 0.99))

    def test_positive_mean_reduces_var(self):
        self.assertLess(parametric_var(0.001, 0.01, 0.99),
                        parametric_var(0.0, 0.01, 0.99))


class CornishFisherTests(unittest.TestCase):
    def test_reduces_to_normal_when_gaussian(self):
        self.assertAlmostEqual(cornish_fisher_var(0.0, 0.01, 0.0, 0.0, 0.99),
                               parametric_var(0.0, 0.01, 0.99), places=12)

    def test_fat_tails_raise_var(self):
        # Positive excess kurtosis must push 99% VaR above the normal answer.
        self.assertGreater(cornish_fisher_var(0.0, 0.01, 0.0, 3.0, 0.99),
                           parametric_var(0.0, 0.01, 0.99))

    def test_negative_skew_raises_var(self):
        self.assertGreater(cornish_fisher_var(0.0, 0.01, -1.0, 0.0, 0.99),
                           cornish_fisher_var(0.0, 0.01, 0.0, 0.0, 0.99))


class MonteCarloTests(unittest.TestCase):
    def test_deterministic_per_seed_and_close_to_analytic(self):
        cov = np.array([[0.0001, 0.00002], [0.00002, 0.00025]])
        w = np.array([0.6, 0.4])
        v1, e1 = monte_carlo_var_es(np.zeros(2), cov, w, 0.99, 50_000, seed=7)
        v2, e2 = monte_carlo_var_es(np.zeros(2), cov, w, 0.99, 50_000, seed=7)
        self.assertEqual((v1, e1), (v2, e2))
        # With normal simulation, MC VaR should approximate delta-normal VaR.
        sigma_p = float(np.sqrt(w @ cov @ w))
        self.assertAlmostEqual(v1, parametric_var(0.0, sigma_p, 0.99), delta=0.15 * sigma_p)
        self.assertGreater(e1, v1)


class SummaryTests(unittest.TestCase):
    def test_all_methods_present_and_positive(self):
        rng = np.random.default_rng(3)
        R = rng.multivariate_normal([0.0003, 0.0001],
                                    [[0.0001, 0.00001], [0.00001, 0.00004]], size=1500)
        s = var_summary(R, np.array([0.5, 0.5]), alpha=0.99)
        d = s.as_dict()
        self.assertEqual(len(d), 7)
        for name, val in d.items():
            self.assertGreater(val, 0.0, name)
        self.assertGreaterEqual(d["Historical ES"], d["Historical VaR"])
        self.assertGreaterEqual(d["Monte Carlo ES"], d["Monte Carlo VaR"])


if __name__ == "__main__":
    unittest.main()
