"""Phase 2 constants, contract schemas, and default operational parameters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT_DIR / "models"

RATES_JSON_PATH = DATA_DIR / "phase2_rates.json"
CAPACITY_JSON_PATH = DATA_DIR / "phase2_capacity.json"

# Least-squares fixed rates calibrated on 2016-2023 onsite data (fallback defaults)
DEFAULT_RESOURCE_RATES: Dict[str, float] = {
    "saline": 2.384,
    "kits": 1.158,
    "beds": 0.2955,
}

# Dataset resource column names
RESOURCE_COLS = (
    "saline_bags_needed",
    "diagnostic_kits_needed",
    "hospital_beds_required",
)

# Priority tier weights (Low -> Critical)
TIER_WEIGHTS: Dict[str, float] = {
    "Low": 1.0,
    "Watch": 2.0,
    "High": 4.0,
    "Critical": 8.0,
}

# Hospital occupancy thresholds (Bagust et al., BMJ 1999)
BED_SURGE_THRESHOLD = 0.85
BED_OVERFLOW_THRESHOLD = 1.00
MIN_WARD_BED_CAPACITY = 2
EXPECTED_TOTAL_BED_CAPACITY = 173

# Numeric precision / ceil safety epsilon (prevents float32 vs float64 1-unit drift)
CEIL_EPSILON = 1e-6
FORECAST_DECIMALS = 3

# Demo & default operational settings
DEFAULT_DEMO_DATE = "2025-06-09"
DEFAULT_HORIZON = 3
DEFAULT_CITY = "All"
DEFAULT_STOCK_PCT = 60.0
DEFAULT_BUFFER_PCT = 0.0
DEFAULT_TOTAL_CREWS = 8
DEFAULT_EFFICACY = 0.30
DEFAULT_RESERVE_BEDS = 0
DEFAULT_CAPACITY_MULTIPLIER = 1.0

# Contract column schemas (only add columns; never rename or remove)
WARD_COLS = (
    "ward_id",
    "city",
    "tier",
    "pred",
    "high_pred",
    "demand_basis",
    "population_density_sqkm",
    "priority_weight",
    "saline_required",
    "saline_allocated",
    "saline_deficit",
    "saline_fill_pct",
    "kits_required",
    "kits_allocated",
    "kits_deficit",
    "kits_fill_pct",
    "beds_required",
    "bed_capacity",
    "bed_utilization_pct",
    "bed_status",
    "bed_overflow",
    "bed_net_deficit",
    "crews_needed",
    "crews_assigned",
    "crews_deficit",
    "cases_prevented",
    "beds_prevented",
    "saline_prevented",
    "roi_per_crew",
    "density_crew_score",
    "deficit_flag",
    "red_alert",
    "code_black",
)

TRANSFER_COLS = (
    "city",
    "from_ward",
    "to_ward",
    "patients_moved",
    "distance_km",
)

CITY_COLS = (
    "city",
    "total_wards",
    "beds_required",
    "bed_capacity",
    "beds_routed",
    "beds_unmet",
    "utilization_pct",
    "remaining_headroom",
    "city_pool_exhausted",
    "saline_required",
    "saline_allocated",
    "saline_deficit",
    "kits_required",
    "kits_allocated",
    "kits_deficit",
    "crews_assigned",
    "red_alert_wards",
    "code_black_wards",
)


def load_resource_rates() -> Dict[str, float]:
    """Load calibrated rates from JSON, falling back to verified defaults."""
    if RATES_JSON_PATH.exists():
        with RATES_JSON_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rates = data.get("rates", data)
        return {
            "saline": float(rates["saline"]),
            "kits": float(rates["kits"]),
            "beds": float(rates["beds"]),
        }
    return dict(DEFAULT_RESOURCE_RATES)


def load_bed_capacities() -> Dict[str, int]:
    """Load ward-level P90 bed capacities from JSON."""
    if CAPACITY_JSON_PATH.exists():
        with CAPACITY_JSON_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        caps = data.get("capacities", data)
        return {str(k): int(v) for k, v in caps.items()}
    return {}