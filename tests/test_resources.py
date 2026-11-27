"""Unit and invariant tests for R1 Constrained Resource Allocation (dengue/resources.py)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dengue.command import CommandParams, run_command_center
from dengue.resources import build_procurement_table, water_fill_city


class TestResourcesAllocation(unittest.TestCase):
    def _run(self, **kwargs):
        base = dict(
            issue_date="2025-06-09",
            horizon=3,
            city="All",
            stock_pct=60.0,
            buffer_pct=0.0,
            overrides=None,
            total_crews=8,
            efficacy=0.30,
            reserve_beds=0,
            offline=True,
            timestamp="2026-11-27T12:00:00Z",
        )
        base.update(kwargs)
        return run_command_center(
            CommandParams(**base),
            use_real={"resources": True, "routing": False, "simulator": False},
        )

    def test_conservation_and_deficit_invariants_across_stock_levels(self) -> None:
        for stock in (30.0, 50.0, 60.0, 80.0, 100.0):
            with self.subTest(stock_pct=stock):
                res = self._run(stock_pct=stock)
                w = res.wards

                # allocated + deficit == required
                self.assertTrue((w["saline_allocated"] + w["saline_deficit"] == w["saline_required"]).all())
                self.assertTrue((w["kits_allocated"] + w["kits_deficit"] == w["kits_required"]).all())

                # Non-negative and bounded
                self.assertTrue((w["saline_allocated"] >= 0).all())
                self.assertTrue((w["saline_allocated"] <= w["saline_required"]).all())
                self.assertTrue((w["kits_allocated"] >= 0).all())
                self.assertTrue((w["kits_allocated"] <= w["kits_required"]).all())

                # Per-city stock conservation: sum(allocated) <= city_stock
                for _, grp in w.groupby("city"):
                    city_sal_stock = (
                        int(grp["saline_required"].sum())
                        if stock >= 100.0
                        else int(round(grp["saline_required"].sum() * (stock / 100.0)))
                    )
                    self.assertLessEqual(int(grp["saline_allocated"].sum()), city_sal_stock)

                # 100% stock -> 0 consumable deficit
                if stock == 100.0:
                    self.assertEqual(int(w["saline_deficit"].sum()), 0)
                    self.assertEqual(int(w["kits_deficit"].sum()), 0)

    def test_priority_monotonicity_and_no_starvation(self) -> None:
        # Synthetic single-city stress test: 6 wards with distinct weights, 50% stock
        demands = np.array([100, 100, 100, 100, 100, 100], dtype=int)
        weights = np.array([16.0, 12.0, 8.0, 6.0, 3.0, 1.5], dtype="float64")
        ward_ids = [f"W{i}" for i in range(6)]
        stock = 300  # 50%

        alloc = water_fill_city(demands, weights, ward_ids, available_stock=stock, stock_pct=50.0)
        self.assertEqual(int(alloc.sum()), stock)
        # No starvation (every ward >= 25% adaptive floor = 25 units)
        self.assertTrue((alloc >= 25).all())
        # Higher weight never receives lower fill than lower weight
        fills = alloc / demands
        for i in range(len(fills) - 1):
            self.assertGreaterEqual(fills[i], fills[i + 1])

    def test_ultra_low_stock_risk_12_guard(self) -> None:
        # Available stock (3) < active wards (6): top 3 weights get 1 unit, sum == 3
        demands = np.array([20, 15, 10, 10, 10, 10], dtype=int)
        weights = np.array([1.0, 8.0, 4.0, 12.0, 2.0, 6.0], dtype="float64")
        ward_ids = [f"W{i}" for i in range(6)]

        alloc = water_fill_city(demands, weights, ward_ids, available_stock=3, stock_pct=5.0)
        self.assertEqual(int(alloc.sum()), 3)
        # Top 3 weights are indices 3 (12.0), 1 (8.0), 5 (6.0)
        self.assertEqual(alloc.tolist(), [0, 1, 0, 1, 0, 1])

    def test_safety_buffer_stress_test_keeps_allocated_constant(self) -> None:
        r0 = self._run(stock_pct=60.0, buffer_pct=0.0)
        r20 = self._run(stock_pct=60.0, buffer_pct=20.0)

        # Total allocated stays identical; required and deficit increase
        self.assertEqual(
            int(r0.wards["saline_allocated"].sum()),
            int(r20.wards["saline_allocated"].sum()),
        )
        self.assertGreater(
            int(r20.wards["saline_required"].sum()),
            int(r0.wards["saline_required"].sum()),
        )
        self.assertGreater(
            int(r20.wards["saline_deficit"].sum()),
            int(r0.wards["saline_deficit"].sum()),
        )

    def test_resource_override_panel_precedence(self) -> None:
        r_main = self._run(stock_pct=60.0, overrides=None)
        r_ovr = self._run(stock_pct=60.0, overrides={"saline": 100.0})

        # Saline overridden to 100% -> 0 saline deficit; kits remain identical to main 60%
        self.assertEqual(int(r_ovr.wards["saline_deficit"].sum()), 0)
        pd.testing.assert_series_equal(r_main.wards["kits_allocated"], r_ovr.wards["kits_allocated"])

    def test_spraying_crews_integer_and_bounded(self) -> None:
        for crews in (0, 5, 8, 25):
            with self.subTest(total_crews=crews):
                res = self._run(total_crews=crews)
                c = res.wards["crews_assigned"]
                self.assertTrue(np.issubdtype(c.dtype, np.integer))
                self.assertTrue((c >= 0).all())
                self.assertLessEqual(int(c.sum()), crews)
                if crews == 0:
                    self.assertEqual(int(c.sum()), 0)

    def test_procurement_table_generation(self) -> None:
        res = self._run(stock_pct=60.0)
        proc = build_procurement_table(res.wards, res.metadata["rates"])
        self.assertGreater(len(proc), 0)
        self.assertTrue({"city", "order_today_saline", "order_today_kits", "projected_beds_w4"}.issubset(proc.columns))


if __name__ == "__main__":
    unittest.main()