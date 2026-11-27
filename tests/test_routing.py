"""Unit and invariant tests for R2 Hospital Surge & Intra-City MILP Routing."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dengue.command import CommandParams, run_command_center
from dengue.routing import route_hospital_overflow


class TestHospitalRouting(unittest.TestCase):
    def _run(self, **kwargs):
        base = dict(
            issue_date="2025-06-09",
            horizon=3,
            city="All",
            stock_pct=60.0,
            buffer_pct=0.0,
            total_crews=8,
            efficacy=0.30,
            reserve_beds=0,
            offline=True,
            timestamp="2026-11-27T12:00:00Z",
        )
        base.update(kwargs)
        return run_command_center(
            CommandParams(**base),
            use_real={"resources": True, "routing": True, "simulator": False},
        )

    def test_routing_invariants_real_forecast(self) -> None:
        for reserve in (0, 1):
            for h in (2, 3, 4):
                with self.subTest(reserve_beds=reserve, horizon=h):
                    res = self._run(reserve_beds=reserve, horizon=h)
                    w = res.wards
                    t = res.transfers
                    c = res.cities

                    ward_to_city = dict(zip(w["ward_id"], w["city"]))
                    ward_cap = dict(zip(w["ward_id"], w["bed_capacity"]))
                    ward_req = dict(zip(w["ward_id"], w["beds_required"]))

                    # 1. All transfers strictly intra-city & positive integers
                    if not t.empty:
                        self.assertTrue(np.issubdtype(t["patients_moved"].dtype, np.integer))
                        self.assertTrue((t["patients_moved"] > 0).all())
                        self.assertTrue((t["distance_km"] >= 0.0).all())
                        for _, row in t.iterrows():
                            self.assertEqual(ward_to_city[row["from_ward"]], row["city"])
                            self.assertEqual(ward_to_city[row["to_ward"]], row["city"])
                            self.assertNotEqual(row["from_ward"], row["to_ward"])

                    # 2. Receiver capacity (minus reserve_beds) is never exceeded
                    rec_in = t.groupby("to_ward")["patients_moved"].sum().to_dict() if not t.empty else {}
                    for wid, moved_in in rec_in.items():
                        self.assertLessEqual(
                            ward_req[wid] + int(moved_in),
                            ward_cap[wid] - reserve,
                        )

                    # 3. Ward-level & city-level conservation: moved + unmet == overflow
                    self.assertTrue(
                        (w["beds_routed_out"] + w["bed_net_deficit"] == w["bed_overflow"]).all()
                    )
                    for _, crow in c.iterrows():
                        city_ov = int(w.loc[w["city"] == crow["city"], "bed_overflow"].sum())
                        self.assertEqual(int(crow["beds_routed"]) + int(crow["beds_unmet"]), city_ov)
                        self.assertEqual(bool(crow["city_pool_exhausted"]), int(crow["beds_unmet"]) > 0)

    def test_milp_prefers_nearest_receiver_and_respects_reserve_bed(self) -> None:
        # Synthetic single-city network:
        # W1 overflows by 3 beds; W2 is 1 km away (cap 5, req 2 -> 3 spare); W3 is 20 km away (cap 5, req 2 -> 3 spare)
        wards = pd.DataFrame(
            [
                {
                    "ward_id": "W1", "city": "Dhaka", "tier": "Critical",
                    "beds_required": 6, "bed_capacity": 3,
                    "saline_required": 10, "saline_allocated": 6, "saline_deficit": 4,
                    "kits_required": 5, "kits_allocated": 5, "kits_deficit": 0,
                    "crews_assigned": 2, "crews_deficit": 0,
                },
                {
                    "ward_id": "W2", "city": "Dhaka", "tier": "Watch",
                    "beds_required": 2, "bed_capacity": 5,
                    "saline_required": 4, "saline_allocated": 4, "saline_deficit": 0,
                    "kits_required": 2, "kits_allocated": 2, "kits_deficit": 0,
                    "crews_assigned": 0, "crews_deficit": 0,
                },
                {
                    "ward_id": "W3", "city": "Dhaka", "tier": "Low",
                    "beds_required": 2, "bed_capacity": 5,
                    "saline_required": 2, "saline_allocated": 2, "saline_deficit": 0,
                    "kits_required": 1, "kits_allocated": 1, "kits_deficit": 0,
                    "crews_assigned": 0, "crews_deficit": 0,
                },
            ]
        )
        coords = {
            "W1": (23.800, 90.400),
            "W2": (23.809, 90.400),  # ~1.0 km north
            "W3": (23.980, 90.400),  # ~20.0 km north
        }

        # reserve_beds=0 -> all 3 patients route to nearest W2
        w0, t0, c0, _ = route_hospital_overflow(wards, CommandParams(reserve_beds=0), coords_override=coords)
        self.assertEqual(len(t0), 1)
        self.assertEqual(t0.iloc[0]["to_ward"], "W2")
        self.assertEqual(int(t0.iloc[0]["patients_moved"]), 3)
        self.assertEqual(int(w0.loc[w0["ward_id"] == "W1", "bed_net_deficit"].iloc[0]), 0)
        self.assertFalse(bool(c0.iloc[0]["city_pool_exhausted"]))

        # reserve_beds=1 -> W2 can only take 2 beds, so 3rd patient routes to W3
        w1, t1, c1, _ = route_hospital_overflow(wards, CommandParams(reserve_beds=1), coords_override=coords)
        moved_map = dict(zip(t1["to_ward"], t1["patients_moved"]))
        self.assertEqual(moved_map, {"W2": 2, "W3": 1})
        self.assertEqual(int(w1.loc[w1["ward_id"] == "W1", "bed_net_deficit"].iloc[0]), 0)
        self.assertFalse(bool(c1.iloc[0]["city_pool_exhausted"]))

    def test_surge_bands_and_exhausted_escalation(self) -> None:
        # High capacity multiplier -> no exhaustion; tight capacity -> exhaustion alert fires
        r_ample = self._run(capacity_multiplier=5.0)
        self.assertEqual(int(r_ample.cities["beds_unmet"].sum()), 0)
        self.assertFalse(r_ample.cities["city_pool_exhausted"].any())

        r_tight = self._run(capacity_multiplier=0.25)
        self.assertGreater(int(r_tight.cities["beds_unmet"].sum()), 0)
        self.assertTrue(r_tight.cities["city_pool_exhausted"].any())
        levels = [a["level"] for a in r_tight.alerts]
        self.assertIn("CITY_EXHAUSTED", levels)

    def test_deterministic_routing_repeatability(self) -> None:
        r1 = self._run(horizon=3, reserve_beds=0)
        r2 = self._run(horizon=3, reserve_beds=0)
        pd.testing.assert_frame_equal(r1.wards, r2.wards)
        pd.testing.assert_frame_equal(r1.transfers, r2.transfers)
        pd.testing.assert_frame_equal(r1.cities, r2.cities)


if __name__ == "__main__":
    unittest.main()