"""Contract-first orchestrator for AlertAhead Phase 2 Command Center."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from dengue.fallback import (
    get_forecast,
    get_git_short_hash,
    load_predictions_parquet,
    load_ward_metadata,
    safe_ceil,
)
from dengue.phase2_config import (
    BED_OVERFLOW_THRESHOLD,
    BED_SURGE_THRESHOLD,
    CITY_COLS,
    DEFAULT_BUFFER_PCT,
    DEFAULT_CAPACITY_MULTIPLIER,
    DEFAULT_CITY,
    DEFAULT_DEMO_DATE,
    DEFAULT_EFFICACY,
    DEFAULT_HORIZON,
    DEFAULT_RESERVE_BEDS,
    DEFAULT_STOCK_PCT,
    DEFAULT_TOTAL_CREWS,
    MIN_WARD_BED_CAPACITY,
    TIER_WEIGHTS,
    TRANSFER_COLS,
    WARD_COLS,
    load_bed_capacities,
    load_resource_rates,
)

# Flip each flag to True as its real backend module is verified and tested.
USE_REAL: Dict[str, bool] = {
    "resources": True,
    "routing": False,
    "simulator": False,
}


@dataclass(frozen=True)
class CommandParams:
    issue_date: str = DEFAULT_DEMO_DATE
    horizon: int = DEFAULT_HORIZON
    city: str = DEFAULT_CITY
    stock_pct: float = DEFAULT_STOCK_PCT
    buffer_pct: float = DEFAULT_BUFFER_PCT
    overrides: Optional[Dict[str, float]] = None
    total_crews: int = DEFAULT_TOTAL_CREWS
    efficacy: float = DEFAULT_EFFICACY
    reserve_beds: int = DEFAULT_RESERVE_BEDS
    offline: bool = False
    capacity_multiplier: float = DEFAULT_CAPACITY_MULTIPLIER
    capacity_overrides: Optional[Dict[str, int]] = None
    timestamp: Optional[str] = None


@dataclass
class CommandResult:
    wards: pd.DataFrame
    transfers: pd.DataFrame
    cities: pd.DataFrame
    alerts: List[Dict[str, Any]]
    source: str
    params: CommandParams
    metadata: Dict[str, Any] = field(default_factory=dict)


def _assign_tier(pred_series: pd.Series) -> pd.Series:
    """Assign deterministic Watch/Risk tier using Phase-1 helper if available."""
    try:
        from dengue.engine import watch_tier

        return pred_series.apply(lambda v: str(watch_tier(float(v))))
    except Exception:
        vals = pred_series.astype("float64")
        conditions = [vals >= 20.0, vals >= 10.0, vals >= 4.0]
        choices = ["Critical", "High", "Watch"]
        return pd.Series(np.select(conditions, choices, default="Low"), index=pred_series.index)


def _normalize_tier_label(label: str) -> str:
    low = str(label).strip().lower()
    if "crit" in low or "severe" in low or "red" in low:
        return "Critical"
    if "high" in low or "elev" in low or "orange" in low:
        return "High"
    if "watch" in low or "mod" in low or "yellow" in low:
        return "Watch"
    return "Low"


def _attach_ward_traits(
    fc: pd.DataFrame,
    raw_df: pd.DataFrame,
    issue_date: str,
    capacities: Dict[str, int],
    params: CommandParams,
) -> pd.DataFrame:
    df = fc.copy()
    issue_ts = pd.Timestamp(issue_date)

    if not raw_df.empty:
        latest_traits = (
            raw_df.sort_values("date")
            .groupby("ward_id", as_index=False)
            .last()[["ward_id"] + [c for c in ("population_density_sqkm",) if c in raw_df.columns]]
        )
        df = df.merge(latest_traits, on="ward_id", how="left")

        if "date" in raw_df.columns and "cases_actual" in raw_df.columns:
            now_sub = raw_df.loc[raw_df["date"] == issue_ts, ["ward_id", "cases_actual"]].rename(
                columns={"cases_actual": "cases_now"}
            )
            df = df.merge(now_sub, on="ward_id", how="left")

    if "population_density_sqkm" not in df.columns:
        df["population_density_sqkm"] = 10000.0
    df["population_density_sqkm"] = df["population_density_sqkm"].fillna(10000.0).astype("float64")

    if "cases_now" not in df.columns:
        df["cases_now"] = np.nan

    cap_edits = params.capacity_overrides or {}
    mult = max(0.25, float(params.capacity_multiplier))
    base_caps = [
        int(cap_edits.get(wid, capacities.get(wid, MIN_WARD_BED_CAPACITY)))
        for wid in df["ward_id"]
    ]
    df["bed_capacity"] = [max(1, int(round(c * mult))) for c in base_caps]

    return df


def _summarize_cities_and_alerts(
    df: pd.DataFrame,
    transfers: pd.DataFrame,
) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Rebuild city-level rollups and escalation alerts from the ward DataFrame."""
    city_rows = []
    for city_name, grp in df.groupby("city", sort=True):
        beds_req = int(grp["beds_required"].sum())
        bed_cap = int(grp["bed_capacity"].sum())
        beds_unmet = int(grp["bed_net_deficit"].sum())
        beds_routed = (
            int(transfers.loc[transfers["city"] == city_name, "patients_moved"].sum())
            if not transfers.empty
            else 0
        )
        headroom = max(0, bed_cap - beds_req)
        exhausted = beds_unmet > 0 or beds_req > bed_cap
        city_rows.append(
            {
                "city": str(city_name),
                "total_wards": int(len(grp)),
                "beds_required": beds_req,
                "bed_capacity": bed_cap,
                "beds_routed": beds_routed,
                "beds_unmet": beds_unmet,
                "utilization_pct": round(100.0 * beds_req / bed_cap, 1) if bed_cap > 0 else 0.0,
                "remaining_headroom": headroom,
                "city_pool_exhausted": bool(exhausted),
                "saline_required": int(grp["saline_required"].sum()),
                "saline_allocated": int(grp["saline_allocated"].sum()),
                "saline_deficit": int(grp["saline_deficit"].sum()),
                "kits_required": int(grp["kits_required"].sum()),
                "kits_allocated": int(grp["kits_allocated"].sum()),
                "kits_deficit": int(grp["kits_deficit"].sum()),
                "crews_assigned": int(grp["crews_assigned"].sum()),
                "red_alert_wards": int(grp["red_alert"].sum()),
                "code_black_wards": int(grp["code_black"].sum()),
            }
        )
    cities = pd.DataFrame(city_rows, columns=list(CITY_COLS))

    alerts: List[Dict[str, Any]] = []
    for _, row in df[df["code_black"]].iterrows():
        alerts.append(
            {
                "level": "CODE_BLACK",
                "city": row["city"],
                "ward_id": row["ward_id"],
                "message": (
                    f"CODE BLACK in {row['ward_id']} ({row['city']}): "
                    f"bed deficit {int(row['bed_net_deficit'])}, saline deficit {int(row['saline_deficit'])}."
                ),
            }
        )
    for _, crow in cities[cities["city_pool_exhausted"]].iterrows():
        alerts.append(
            {
                "level": "CITY_EXHAUSTED",
                "city": crow["city"],
                "ward_id": None,
                "message": (
                    f"City pool exhausted in {crow['city']}: "
                    f"{int(crow['beds_unmet'])} unmet bed(s) — escalate to Civil Surgeon."
                ),
            }
        )
    return cities, alerts


def _run_honest_dummy(
    base_df: pd.DataFrame,
    params: CommandParams,
    rates: Dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """Honest deterministic baseline using parquet forecasts."""
    df = base_df.copy()
    h = int(params.horizon)
    df["pred"] = df[f"pred_h{h}"].astype("float64")
    df["high_pred"] = df[f"high_h{h}"].astype("float64")

    if "tier" not in df.columns:
        df["tier"] = _assign_tier(df["pred"])
    df["tier"] = df["tier"].map(_normalize_tier_label)

    use_upper = df["tier"].isin(["Watch", "High", "Critical"])
    df["demand_basis"] = np.where(use_upper, df["high_pred"], df["pred"]).astype("float64")

    city_max_density = df.groupby("city")["population_density_sqkm"].transform("max").clip(lower=1.0)
    tier_w = df["tier"].map(lambda t: TIER_WEIGHTS.get(t, 1.0)).astype("float64")
    df["priority_weight"] = (tier_w * (1.0 + df["population_density_sqkm"] / city_max_density)).round(4)

    buf_factor = 1.0 + max(0.0, float(params.buffer_pct)) / 100.0
    base_saline = safe_ceil(rates["saline"] * df["demand_basis"])
    base_kits = safe_ceil(rates["kits"] * df["demand_basis"])
    base_beds = safe_ceil(rates["beds"] * df["demand_basis"])

    df["saline_required"] = safe_ceil(base_saline * buf_factor)
    df["kits_required"] = safe_ceil(base_kits * buf_factor)
    df["beds_required"] = safe_ceil(base_beds * buf_factor)

    overrides = params.overrides or {}
    saline_pct = float(overrides.get("saline", params.stock_pct)) / 100.0
    kits_pct = float(overrides.get("kits", params.stock_pct)) / 100.0

    df["saline_allocated"] = np.minimum(
        df["saline_required"], np.floor(base_saline * saline_pct).astype(int)
    )
    df["saline_deficit"] = (df["saline_required"] - df["saline_allocated"]).clip(lower=0)
    df["saline_fill_pct"] = np.where(
        df["saline_required"] > 0,
        (100.0 * df["saline_allocated"] / df["saline_required"]).round(1),
        100.0,
    )

    df["kits_allocated"] = np.minimum(
        df["kits_required"], np.floor(base_kits * kits_pct).astype(int)
    )
    df["kits_deficit"] = (df["kits_required"] - df["kits_allocated"]).clip(lower=0)
    df["kits_fill_pct"] = np.where(
        df["kits_required"] > 0,
        (100.0 * df["kits_allocated"] / df["kits_required"]).round(1),
        100.0,
    )

    util_ratio = np.where(df["bed_capacity"] > 0, df["beds_required"] / df["bed_capacity"], 0.0)
    df["bed_utilization_pct"] = (100.0 * util_ratio).round(1)
    df["bed_status"] = np.select(
        [util_ratio > BED_OVERFLOW_THRESHOLD, util_ratio >= BED_SURGE_THRESHOLD],
        ["Overflow", "Surge Warning"],
        default="Normal",
    )
    df["bed_overflow"] = np.maximum(0, df["beds_required"] - df["bed_capacity"]).astype(int)
    df["bed_net_deficit"] = df["bed_overflow"].astype(int)

    df["density_crew_score"] = (df["pred"] * df["priority_weight"]).round(4)
    df["crews_needed"] = np.where(
        df["tier"].isin(["High", "Critical"]), 2, np.where(df["pred"] >= 5.0, 1, 0)
    ).astype(int)
    df["crews_assigned"] = 0
    total_crews = max(0, int(params.total_crews))
    if total_crews > 0 and len(df) > 0:
        ranked_idx = df.sort_values(
            ["density_crew_score", "ward_id"], ascending=[False, True], kind="mergesort"
        ).index.tolist()
        for i in range(total_crews):
            df.loc[ranked_idx[i % len(ranked_idx)], "crews_assigned"] += 1
    df["crews_deficit"] = np.maximum(0, df["crews_needed"] - df["crews_assigned"]).astype(int)

    df["cases_prevented"] = 0.0
    df["beds_prevented"] = 0.0
    df["saline_prevented"] = 0.0
    df["roi_per_crew"] = 0.0

    df["deficit_flag"] = (
        (df["saline_deficit"] > 0)
        | (df["kits_deficit"] > 0)
        | (df["bed_net_deficit"] > 0)
        | (df["crews_deficit"] > 0)
    )
    df["red_alert"] = df["tier"].isin(["High", "Critical"]) & (
        (df["saline_deficit"] > 0) | (df["kits_deficit"] > 0) | (df["bed_net_deficit"] > 0)
    )
    df["code_black"] = (df["bed_net_deficit"] > 0) & (df["saline_deficit"] > 0)

    transfers = pd.DataFrame(columns=list(TRANSFER_COLS))
    df = df.sort_values("ward_id", kind="mergesort").reset_index(drop=True)
    cities, alerts = _summarize_cities_and_alerts(df, transfers)
    return df, transfers, cities, alerts


def _resolve_flags(use_real: Optional[bool | Dict[str, bool]]) -> Dict[str, bool]:
    flags = dict(USE_REAL)
    if isinstance(use_real, bool):
        return {k: use_real for k in flags}
    if isinstance(use_real, dict):
        for k, v in use_real.items():
            if k in flags:
                flags[k] = bool(v)
    return flags


def run_command_center(
    params: Optional[CommandParams] = None,
    use_real: Optional[bool | Dict[str, bool]] = None,
    preds_df: Optional[pd.DataFrame] = None,
    raw_df: Optional[pd.DataFrame] = None,
) -> CommandResult:
    """Single orchestrator entrypoint for UI, make_exports.py, and all tests."""
    params = params or CommandParams()
    flags = _resolve_flags(use_real)

    rates = load_resource_rates()
    capacities = load_bed_capacities()
    preds = preds_df if preds_df is not None else load_predictions_parquet()
    meta = raw_df if raw_df is not None else load_ward_metadata()

    fc, source = get_forecast(preds, params.issue_date, params.horizon, raw_df=meta)
    base_df = _attach_ward_traits(fc, meta, params.issue_date, capacities, params)

    if params.city and params.city != "All":
        base_df = base_df.loc[base_df["city"] == params.city].copy().reset_index(drop=True)

    wards, transfers, cities, alerts = _run_honest_dummy(base_df, params, rates)

    sim_output = None
    if flags["simulator"]:
        try:
            from dengue import simulator as real_sim

            sim_output = real_sim.run_simulator(wards, params, rates)
            wards = real_sim.attach_simulator_columns(wards, sim_output)
        except ImportError:
            pass

    if flags["resources"]:
        try:
            from dengue import resources as real_res

            wards = real_res.allocate_resources(
                wards,
                params=params,
                rates=rates,
                sim_output=sim_output,
            )
            cities, alerts = _summarize_cities_and_alerts(wards, transfers)
        except ImportError:
            pass

    if flags["routing"]:
        try:
            from dengue import routing as real_rt

            wards, transfers, cities, alerts = real_rt.route_hospital_overflow(
                wards,
                params=params,
            )
        except ImportError:
            pass

    for col in WARD_COLS:
        if col not in wards.columns:
            raise KeyError(f"Contract violation: missing ward column '{col}'")
    wards = wards.sort_values("ward_id", kind="mergesort").reset_index(drop=True)

    metadata = {
        "git_hash": get_git_short_hash(),
        "offline": bool(params.offline),
        "flags": flags,
        "rates": rates,
        "timestamp": params.timestamp,
    }
    return CommandResult(
        wards=wards,
        transfers=transfers,
        cities=cities,
        alerts=alerts,
        source=source,
        params=params,
        metadata=metadata,
    )