"""R1: Constrained Resource Allocation Engine (Weighted Water-Filling + Adaptive Floor)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from dengue.fallback import safe_ceil
from dengue.phase2_config import (
    BED_OVERFLOW_THRESHOLD,
    BED_SURGE_THRESHOLD,
    TIER_WEIGHTS,
)


def compute_priority_weights(
    df: pd.DataFrame,
    use_kits_trend: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute city-normalized priority weights for saline/general and diagnostic kits.

    Formula: tier_weight (1/2/4/8) * (1 + density / city_max_density).
    Optional kits trend boost adds +0.25 multiplier when pred_h4 > cases_now.
    """
    city_max_density = (
        df.groupby("city")["population_density_sqkm"].transform("max").clip(lower=1.0).to_numpy(dtype="float64")
    )
    density = df["population_density_sqkm"].to_numpy(dtype="float64")
    tier_w = df["tier"].map(lambda t: TIER_WEIGHTS.get(str(t), 1.0)).to_numpy(dtype="float64")

    base_weight = tier_w * (1.0 + density / city_max_density)
    kits_weight = base_weight.copy()

    if use_kits_trend and "pred_h4" in df.columns and "cases_now" in df.columns:
        pred_h4 = pd.to_numeric(df["pred_h4"], errors="coerce").to_numpy(dtype="float64")
        cases_now = pd.to_numeric(df["cases_now"], errors="coerce").to_numpy(dtype="float64")
        rising = np.isfinite(pred_h4) & np.isfinite(cases_now) & (pred_h4 > cases_now)
        kits_weight = np.where(rising, kits_weight * 1.25, kits_weight)

    return np.round(base_weight, 6), np.round(kits_weight, 6)


def water_fill_city(
    demands: np.ndarray,
    weights: np.ndarray,
    ward_ids: list[str],
    available_stock: int,
    stock_pct: float,
) -> np.ndarray:
    """Weighted water-filling with adaptive floor and largest-remainder integer rounding.

    Guarantees:
      - sum(allocated) == min(available_stock, sum(demands)) <= available_stock
      - 0 <= allocated_i <= demands_i
      - Adaptive floor = min(25%, stock_pct / 2), preventing starvation at low stock
        while preserving priority gradient (0 cliff within same tier).
      - Risk #12 guard: if available_stock < active_wards, gives 1 unit in priority order.
    """
    d = np.asarray(demands, dtype=int)
    w = np.asarray(weights, dtype="float64")
    n = len(d)
    alloc = np.zeros(n, dtype=int)

    stock = max(0, int(available_stock))
    total_demand = int(d.sum())
    if n == 0 or stock <= 0 or total_demand <= 0:
        return alloc
    if stock >= total_demand:
        return d.copy()

    active = np.flatnonzero(d > 0)
    m = len(active)

    # Risk #12: ultra-low stock (< number of wards with positive demand)
    if stock <= m:
        ranked = sorted(active.tolist(), key=lambda i: (-w[i], ward_ids[i]))
        for idx in ranked[:stock]:
            alloc[idx] = 1
        return alloc

    d_act = d[active].astype("float64")
    w_act = np.maximum(w[active], 1e-9)

    # Adaptive floor: min(25%, half of stock%), capped so floor never exceeds stock
    raw_floor = min(0.25, max(0.0, float(stock_pct) / 100.0) * 0.5)

    def _eval_floor_only(fl: float) -> np.ndarray:
        return np.minimum(d_act, np.maximum(1.0, d_act * fl))

    # Ensure baseline floor + 1-unit minimum does not exceed available stock
    if float(_eval_floor_only(raw_floor).sum()) > stock:
        lo_f, hi_f = 0.0, raw_floor
        for _ in range(50):
            mid_f = 0.5 * (lo_f + hi_f)
            if float(_eval_floor_only(mid_f).sum()) <= stock:
                lo_f = mid_f
            else:
                hi_f = mid_f
        eff_floor = lo_f
        cont = _eval_floor_only(eff_floor)
    else:
        eff_floor = raw_floor

        def _eval_lambda(lam: float) -> np.ndarray:
            fill = np.maximum(eff_floor, np.minimum(1.0, lam * w_act))
            return np.minimum(d_act, np.maximum(1.0, d_act * fill))

        lo = 0.0
        hi = 1.0 / float(w_act.min())
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if float(_eval_lambda(mid).sum()) < stock:
                lo = mid
            else:
                hi = mid
        cont = _eval_lambda(0.5 * (lo + hi))

    # Largest-remainder rounding with deterministic tie-break (-remainder, -weight, ward_id)
    floors = np.floor(cont).astype(int)
    floors = np.minimum(floors, d[active])
    rem = stock - int(floors.sum())

    if rem > 0:
        fracs = cont - floors
        eligible = [k for k in range(m) if floors[k] < d[active[k]]]
        eligible.sort(key=lambda k: (-fracs[k], -w_act[k], ward_ids[active[k]]))
        for k in eligible[:rem]:
            floors[k] += 1
    elif rem < 0:
        # Safety guard (should not occur, enforces strict sum <= stock)
        excess = -rem
        eligible = [k for k in range(m) if floors[k] > 0]
        eligible.sort(key=lambda k: (cont[k] - floors[k], w_act[k], ward_ids[active[k]]))
        for k in eligible[:excess]:
            floors[k] -= 1

    alloc[active] = floors
    return alloc


def _allocate_consumable_by_city(
    df: pd.DataFrame,
    base_demand: np.ndarray,
    required_demand: np.ndarray,
    weights: np.ndarray,
    stock_pct: float,
) -> np.ndarray:
    """Allocate a consumable resource independently within each city pool.

    Stock is computed from base_demand (never buffered demand), so increasing
    Safety Buffer % keeps total allocated constant and stress-tests deficits.
    """
    alloc = np.zeros(len(df), dtype=int)
    pct = max(0.0, min(100.0, float(stock_pct)))

    for _, grp in df.groupby("city", sort=True):
        idx = grp.index.to_numpy()
        city_base_total = int(base_demand[idx].sum())
        if pct >= 100.0:
            city_stock = city_base_total
        else:
            city_stock = int(round(city_base_total * (pct / 100.0)))

        city_alloc = water_fill_city(
            demands=required_demand[idx],
            weights=weights[idx],
            ward_ids=grp["ward_id"].astype(str).tolist(),
            available_stock=city_stock,
            stock_pct=pct,
        )
        alloc[idx] = city_alloc

    return alloc


def allocate_spraying_crews(
    df: pd.DataFrame,
    total_crews: int,
    sim_output: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concentrated hotspot allocation for vector-control spraying crews (no floor).

    Returns (crews_needed, crews_assigned, density_crew_score).
    Uses marginal ROI from simulator when available; falls back deterministically
    to predicted_cases * tier_weight * (1 + density / city_max_density).
    """
    n = len(df)
    ward_ids = df["ward_id"].astype(str).tolist()
    city_max_density = (
        df.groupby("city")["population_density_sqkm"].transform("max").clip(lower=1.0).to_numpy(dtype="float64")
    )
    density_mult = 1.0 + df["population_density_sqkm"].to_numpy(dtype="float64") / city_max_density

    total = max(0, int(total_crews))
    assigned = np.zeros(n, dtype=int)

    # Case 1: Simulator marginal ROI matrix available (shape: dict[ward_id -> list[float]])
    marginal_roi_map = sim_output.get("marginal_roi") if isinstance(sim_output, dict) else None
    if marginal_roi_map:
        needed = np.zeros(n, dtype=int)
        first_step_score = np.zeros(n, dtype="float64")
        marginal_lists: list[list[float]] = []

        for i, wid in enumerate(ward_ids):
            steps = [max(0.0, float(v)) for v in marginal_roi_map.get(wid, [])]
            marginal_lists.append(steps)
            # Demand = number of crews while marginal ROI >= 1 case/crew
            needed[i] = sum(1 for v in steps if v >= 1.0)
            first_step_score[i] = (steps[0] if steps else 0.0) * density_mult[i]

        for _ in range(total):
            best_idx = -1
            best_key = (-1.0, "")
            for i in range(n):
                k = assigned[i]
                if k < len(marginal_lists[i]):
                    marg = marginal_lists[i][k]
                elif marginal_lists[i]:
                    marg = marginal_lists[i][-1] * (0.5 ** (k - len(marginal_lists[i]) + 1))
                else:
                    marg = 0.0
                score = marg * density_mult[i]
                if score <= 0.0:
                    continue
                if best_idx == -1 or (score > best_key[0]) or (
                    score == best_key[0] and ward_ids[i] < best_key[1]
                ):
                    best_idx = i
                    best_key = (score, ward_ids[i])
            if best_idx == -1:
                break
            assigned[best_idx] += 1

        return needed, assigned, np.round(first_step_score, 4)

    # Case 2: Deterministic heuristic fallback/stub (pred * tier_weight * (1 + density/max))
    pred = df["pred"].to_numpy(dtype="float64")
    tier_w = df["tier"].map(lambda t: TIER_WEIGHTS.get(str(t), 1.0)).to_numpy(dtype="float64")
    base_score = np.maximum(0.0, pred * tier_w * density_mult)

    # Heuristic crew demand: hotspot-focused integer need
    needed = np.where(
        df["tier"].isin(["Critical", "High"]),
        np.clip(safe_ceil(pred / 10.0), 1, 4),
        np.where((df["tier"] == "Watch") & (pred >= 4.0), 1, 0),
    ).astype(int)

    # Greedy hotspot allocation with diminishing marginal return per additional crew
    for _ in range(total):
        best_idx = -1
        best_score = 0.0
        for i in range(n):
            if base_score[i] <= 0.0:
                continue
            marg_score = base_score[i] / (1.0 + 0.75 * assigned[i])
            if marg_score > best_score or (
                marg_score == best_score and best_idx != -1 and ward_ids[i] < ward_ids[best_idx]
            ):
                best_idx = i
                best_score = marg_score
        if best_idx == -1:
            break
        assigned[best_idx] += 1

    return needed, assigned, np.round(base_score, 4)


def allocate_resources(
    wards: pd.DataFrame,
    params: Any,
    rates: Dict[str, float],
    sim_output: Optional[Dict[str, Any]] = None,
    use_kits_trend: bool = False,
) -> pd.DataFrame:
    """Run full R1 constrained resource allocation on the ward DataFrame."""
    df = wards.copy().reset_index(drop=True)

    # High/Critical/Watch use P90 upper range (high_pred); Low uses point forecast
    use_upper = df["tier"].isin(["Watch", "High", "Critical"])
    df["demand_basis"] = np.where(use_upper, df["high_pred"], df["pred"]).astype("float64")

    # Base demand (before Safety Buffer)
    base_saline = safe_ceil(rates["saline"] * df["demand_basis"])
    base_kits = safe_ceil(rates["kits"] * df["demand_basis"])
    base_beds = safe_ceil(rates["beds"] * df["demand_basis"])

    # Safety Buffer % (0-30%, default 0): increases required, keeps stock tied to base_demand
    buffer_pct = max(0.0, float(getattr(params, "buffer_pct", 0.0)))
    if buffer_pct == 0.0:
        req_saline = base_saline.copy()
        req_kits = base_kits.copy()
        req_beds = base_beds.copy()
    else:
        buf_mult = 1.0 + buffer_pct / 100.0
        req_saline = safe_ceil(base_saline * buf_mult)
        req_kits = safe_ceil(base_kits * buf_mult)
        req_beds = safe_ceil(base_beds * buf_mult)

    df["saline_required"] = req_saline
    df["kits_required"] = req_kits
    df["beds_required"] = req_beds

    # Priority weights
    priority_w, kits_w = compute_priority_weights(df, use_kits_trend=use_kits_trend)
    df["priority_weight"] = np.round(priority_w, 4)

    # Stock % with optional resource-wise override panel (default off -> main slider)
    overrides = getattr(params, "overrides", None) or {}
    main_stock_pct = float(getattr(params, "stock_pct", 100.0))
    saline_pct = float(overrides["saline"]) if overrides.get("saline") is not None else main_stock_pct
    kits_pct = float(overrides["kits"]) if overrides.get("kits") is not None else main_stock_pct

    # City-wise weighted water-filling allocation
    df["saline_allocated"] = _allocate_consumable_by_city(
        df, base_saline, req_saline, priority_w, saline_pct
    )
    df["saline_deficit"] = (df["saline_required"] - df["saline_allocated"]).clip(lower=0).astype(int)
    df["saline_fill_pct"] = np.where(
        df["saline_required"] > 0,
        np.round(100.0 * df["saline_allocated"] / df["saline_required"], 1),
        100.0,
    )

    df["kits_allocated"] = _allocate_consumable_by_city(
        df, base_kits, req_kits, kits_w, kits_pct
    )
    df["kits_deficit"] = (df["kits_required"] - df["kits_allocated"]).clip(lower=0).astype(int)
    df["kits_fill_pct"] = np.where(
        df["kits_required"] > 0,
        np.round(100.0 * df["kits_allocated"] / df["kits_required"], 1),
        100.0,
    )

    # Bed status bands (Bagust et al., BMJ 1999) — routing refines bed_net_deficit in R2
    cap = df["bed_capacity"].to_numpy(dtype="float64")
    util = np.where(cap > 0, df["beds_required"].to_numpy(dtype="float64") / cap, 0.0)
    df["bed_utilization_pct"] = np.round(100.0 * util, 1)
    df["bed_status"] = np.select(
        [util > BED_OVERFLOW_THRESHOLD, util >= BED_SURGE_THRESHOLD],
        ["Overflow", "Surge Warning"],
        default="Normal",
    )
    df["bed_overflow"] = np.maximum(0, df["beds_required"] - df["bed_capacity"]).astype(int)
    df["bed_net_deficit"] = df["bed_overflow"].astype(int)

    # Spraying crew allocation (hotspot concentrated, integer, no floor)
    crews_needed, crews_assigned, density_score = allocate_spraying_crews(
        df,
        total_crews=int(getattr(params, "total_crews", 0)),
        sim_output=sim_output,
    )
    df["crews_needed"] = crews_needed
    df["crews_assigned"] = crews_assigned
    df["crews_deficit"] = np.maximum(0, crews_needed - crews_assigned).astype(int)
    df["density_crew_score"] = density_score

    # Operational alert flags
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

    return df


def build_procurement_table(
    wards: pd.DataFrame,
    rates: Dict[str, float],
    preds_h4: Optional[pd.DataFrame] = None,
    buffer_pct: float = 0.0,
) -> pd.DataFrame:
    """4-week lead-time 'order today' procurement summary (bonus operational table)."""
    df = wards[["ward_id", "city", "tier", "pred", "high_pred"]].copy()
    if preds_h4 is not None and "pred_h4" in preds_h4.columns:
        cols = ["ward_id", "pred_h4"] + (["high_h4"] if "high_h4" in preds_h4.columns else [])
        df = df.merge(preds_h4[cols], on="ward_id", how="left")
        h4_point = df["pred_h4"].fillna(df["pred"]).astype("float64")
        h4_high = df["high_h4"].fillna(df["high_pred"]).astype("float64") if "high_h4" in df.columns else h4_point
    else:
        h4_point = df["pred"].astype("float64")
        h4_high = df["high_pred"].astype("float64")

    basis = np.where(df["tier"].isin(["Watch", "High", "Critical"]), h4_high, h4_point)
    mult = 1.0 + max(0.0, float(buffer_pct)) / 100.0
    df["order_today_saline"] = safe_ceil(safe_ceil(rates["saline"] * basis) * mult)
    df["order_today_kits"] = safe_ceil(safe_ceil(rates["kits"] * basis) * mult)
    df["projected_beds_w4"] = safe_ceil(safe_ceil(rates["beds"] * basis) * mult)

    return (
        df.groupby("city", as_index=False)[
            ["order_today_saline", "order_today_kits", "projected_beds_w4"]
        ]
        .sum()
        .sort_values("city")
        .reset_index(drop=True)
    )