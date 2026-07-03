import unittest

import numpy as np

from marketrisk.decompose import component_var


class ComponentVarTests(unittest.TestCase):
    def setUp(self):
        self.cov = np.array([
            [0.00010, 0.00002, 0.00001],
            [0.00002, 0.00025, 0.00003],
            [0.00001, 0.00003, 0.00016],
        ])
        self.w = np.array([0.5, 0.3, 0.2])

    def test_euler_additivity(self):
        # The whole point of component VaR: the pieces sum exactly to the total.
        df = component_var(self.w, self.cov, alpha=0.99)
        self.assertAlmostEqual(df["component_var"].sum(),
                               df.attrs["total_var"], places=12)
        self.assertAlmostEqual(df["pct_of_total"].sum(), 1.0, places=12)

    def test_symmetric_case_splits_equally(self):
        cov = np.full((2, 2), 0.00002)
        np.fill_diagonal(cov, 0.0001)
        df = component_var(np.array([0.5, 0.5]), cov, alpha=0.99, names=["A", "B"])
        self.assertAlmostEqual(df.loc["A", "component_var"],
                               df.loc["B", "component_var"], places=12)

    def test_names_and_columns(self):
        df = component_var(self.w, self.cov, names=["EQ", "RATES", "GOLD"])
        self.assertEqual(list(df.index), ["EQ", "RATES", "GOLD"])
        for col in ("weight", "marginal_var", "component_var", "pct_of_total"):
            self.assertIn(col, df.columns)

    def test_zero_portfolio_vol_raises(self):
        with self.assertRaises(ValueError):
            component_var(np.zeros(3), self.cov)


if __name__ == "__main__":
    unittest.main()
