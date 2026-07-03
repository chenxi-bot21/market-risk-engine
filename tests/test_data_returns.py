import tempfile
import unittest
from pathlib import Path

import numpy as np

from marketrisk.data import load_csv, resolve_prices, synthetic_prices
from marketrisk.returns import log_returns, normalise_weights, portfolio_returns


class SyntheticDataTests(unittest.TestCase):
    def test_shape_positive_and_deterministic(self):
        p1 = synthetic_prices(n_days=300, seed=1)
        p2 = synthetic_prices(n_days=300, seed=1)
        self.assertEqual(p1.shape, (300, 4))
        self.assertTrue((p1 > 0).all().all())
        self.assertTrue(p1.equals(p2))
        self.assertFalse(p1.equals(synthetic_prices(n_days=300, seed=2)))

    def test_correlation_structure_plausible(self):
        # Equity/bond were generated with negative correlation.
        r = log_returns(synthetic_prices(n_days=2000, seed=42))
        self.assertLess(r["EQUITY"].corr(r["BOND"]), 0.1)


class CsvAndResolveTests(unittest.TestCase):
    def test_csv_roundtrip(self):
        p = synthetic_prices(n_days=100, seed=3)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "prices.csv"
            p.to_csv(path)
            loaded = load_csv(str(path))
        self.assertEqual(loaded.shape, p.shape)
        np.testing.assert_allclose(loaded.values, p.values)

    def test_resolve_validation(self):
        with self.assertRaises(ValueError):
            resolve_prices("csv")                    # missing csv_path
        with self.assertRaises(ValueError):
            resolve_prices("yfinance")               # missing tickers
        with self.assertRaises(ValueError):
            resolve_prices("bloomberg")              # unknown source


class ReturnsTests(unittest.TestCase):
    def test_log_returns_shape_and_no_nan(self):
        p = synthetic_prices(n_days=100, seed=4)
        r = log_returns(p)
        self.assertEqual(len(r), 99)
        self.assertFalse(r.isna().any().any())

    def test_weights_normalised_and_validated(self):
        w = normalise_weights([2.0, 2.0, 4.0, 2.0], 4)
        self.assertAlmostEqual(w.sum(), 1.0)
        with self.assertRaises(ValueError):
            normalise_weights([1.0, 1.0], 4)         # wrong length
        with self.assertRaises(ValueError):
            normalise_weights([1.0, -1.0, 0.0, 0.0], 4)  # sums to zero

    def test_portfolio_returns_match_manual_dot(self):
        p = synthetic_prices(n_days=50, seed=6)
        r = log_returns(p)
        w = np.array([0.4, 0.3, 0.2, 0.1])
        port = portfolio_returns(r, w)
        np.testing.assert_allclose(port.values, r.values @ w)


if __name__ == "__main__":
    unittest.main()
