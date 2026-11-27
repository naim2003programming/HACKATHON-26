"""Build data/phase2_rates.json and data/phase2_capacity.json from onsite dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from dengue.phase2_config import (
    CAPACITY_JSON_PATH,
    DATA_DIR,
    EXPECTED_TOTAL_BED_CAPACITY,
    MIN_WARD_BED_CAPACITY,
    RATES_JSON_PATH,
    RAW_DATA_DIR,
    RESOURCE_COLS,
)


def _find_onsite_dataset() -> Path:
    candidates = []
    for folder in (RAW_DATA_DIR, DATA_DIR):
        if not folder.exists():
            continue
        for ext in ("*.csv", "*.xlsx", "*.parquet"):
            candidates.extend(sorted(folder.glob(ext)))

    # Prefer dataset containing all 3 Phase-2 resource columns
    for path in candidates:
        if path.name.startswith("predictions"):
            continue
        try:
            df_head = _read_table(path, nrows=5)
        except Exception:
            continue
        if set(RESOURCE_COLS).issubset(df_head.columns):
            return path

    raise FileNotFoundError(
        "Onsite dataset with columns "
        f"{RESOURCE_COLS} not found under {RAW_DATA_DIR} or {DATA_DIR}."
    )


def _read_table(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows)
    if suffix == ".xlsx":
        return pd.read_excel(path, nrows=nrows)
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        return df.head(nrows) if nrows is not None else df
    raise ValueError(f"Unsupported dataset format: {path}")


def _detect_col(df: pd.DataFrame, options: tuple[str, ...]) -> str:
    for col in options:
        if col in df.columns:
            return col
    raise KeyError(f"None of {options} found in columns: {list(df.columns)}")


def build_assets(dataset_path: Optional[Path] = None) -> tuple[dict, dict]:
    path = dataset_path or _find_onsite_dataset()
    df = _read_table(path)

    date_col = _detect_col(df, ("date", "week_start", "week_date", "ds"))
    ward_col = _detect_col(df, ("ward_id", "ward", "ward_code"))
    cases_col = _detect_col(df, ("cases", "dengue_cases", "weekly_cases", "reported_cases"))

    df[date_col] = pd.to_datetime(df[date_col])
    df["_year"] = df[date_col].dt.year

    train = df[(df["_year"] >= 2016) & (df["_year"] <= 2023)].copy()
    test_2025 = df[df["_year"] == 2025].copy()

    x = train[cases_col].astype("float64").to_numpy()
    denom = float(np.dot(x, x))
    if denom <= 0:
        raise ValueError("Training cases sum of squares must be positive.")

    col_map = {
        "saline": "saline_bags_needed",
        "kits": "diagnostic_kits_needed",
        "beds": "hospital_beds_required",
    }

    rates: dict[str, float] = {}
    verification_2025: dict[str, dict[str, int]] = {}
    for key, col in col_map.items():
        y = train[col].astype("float64").to_numpy()
        rate = float(np.dot(x, y) / denom)
        rates[key] = round(rate, 6)

        if not test_2025.empty:
            pred_2025 = np.rint(rate * test_2025[cases_col].astype("float64").to_numpy()).astype(int)
            actual_2025 = test_2025[col].astype(int).to_numpy()
            exact = int(np.sum(pred_2025 == actual_2025))
            verification_2025[key] = {"exact": exact, "total": int(len(actual_2025))}

    # Ward-level P90 bed capacity on 2016-2023 using fixed method="lower"
    capacities: dict[str, int] = {}
    for ward_id, grp in train.groupby(ward_col, sort=True):
        vals = grp["hospital_beds_required"].astype("float64").to_numpy()
        p90 = int(np.quantile(vals, 0.90, method="lower"))
        capacities[str(ward_id)] = max(MIN_WARD_BED_CAPACITY, p90)

    total_beds = int(sum(capacities.values()))

    rates_payload = {
        "source_file": str(path.name),
        "fit_years": [2016, 2023],
        "method": "least_squares_through_origin",
        "rates": rates,
        "verification_2025": verification_2025,
    }
    capacity_payload = {
        "source_file": str(path.name),
        "fit_years": [2016, 2023],
        "quantile": 0.90,
        "quantile_method": "lower",
        "min_ward_capacity": MIN_WARD_BED_CAPACITY,
        "total_bed_capacity": total_beds,
        "ward_count": len(capacities),
        "capacities": capacities,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RATES_JSON_PATH.write_text(json.dumps(rates_payload, indent=2), encoding="utf-8")
    CAPACITY_JSON_PATH.write_text(json.dumps(capacity_payload, indent=2), encoding="utf-8")

    print(f"Dataset: {path}")
    print(f"Rates: {rates}")
    print(f"2025 exact match check: {verification_2025}")
    print(
        f"Ward capacities saved: {len(capacities)} wards, "
        f"total beds = {total_beds} (expected {EXPECTED_TOTAL_BED_CAPACITY})"
    )
    return rates_payload, capacity_payload


if __name__ == "__main__":
    build_assets()