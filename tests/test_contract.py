"""Contract and data-foundation tests for Phase 2 Command Center."""

from __future__ import annotations

import unittest

import pandas as pd

from dengue.command import CommandParams, CommandResult, run_command_center
from dengue.phase2_config import (
    CAPACITY_JSON_PATH,
    CITY_COLS,
    EXPECTED_TOTAL_BED_CAPACITY,
    TRANSFER_COLS,
    WARD_COLS,
    load_bed_capacities,
)


class TestCommandContract(unittest.TestCase):
    def setUp(self) -> None:
        self.params = CommandParams(
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

    def test_contract_columns_and_types_dummy_and_real(self) -> None:
        for real_mode in (False, True):
            with self.subTest(use_real=real_mode):
                res = run_command_center(self.params, use_real=real_mode)
                self.assertIsInstance(res, CommandResult)
                self.assertIsInstance(res.wards, pd.DataFrame)
                self.assertIsInstance(res.transfers, pd.DataFrame)
                self.assertIsInstance(res.cities, pd.DataFrame)
                self.assertIsInstance(res.alerts, list)
                self.assertIsInstance(res.source, str)
                self.assertGreater(len(res.wards), 0)

                self.assertTrue(set(WARD_COLS).issubset(res.wards.columns))
                self.assertTrue(set(TRANSFER_COLS).issubset(res.transfers.columns))
                self.assertTrue(set(CITY_COLS).issubset(res.cities.columns))

                # Deterministic ordering by ward_id
                ward_ids = res.wards["ward_id"].tolist()
                self.assertEqual(ward_ids, sorted(ward_ids))

    def test_same_number_online_vs_offline(self) -> None:
        p_online = CommandParams(**{**self.params.__dict__, "offline": False})
        p_offline = CommandParams(**{**self.params.__dict__, "offline": True})

        r_on = run_command_center(p_online, use_real=False)
        r_off = run_command_center(p_offline, use_real=False)

        pd.testing.assert_frame_equal(r_on.wards, r_off.wards)
        pd.testing.assert_frame_equal(r_on.cities, r_off.cities)
        self.assertEqual(r_on.source, r_off.source)

    def test_capacity_json_total_173_when_built(self) -> None:
        if not CAPACITY_JSON_PATH.exists():
            self.skipTest("Run python -m dengue.build_phase2_assets first.")
        caps = load_bed_capacities()
        self.assertEqual(len(caps), 60)
        self.assertEqual(sum(caps.values()), EXPECTED_TOTAL_BED_CAPACITY)


if __name__ == "__main__":
    unittest.main()