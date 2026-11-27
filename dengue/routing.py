"""R2: Hospital Surge Band & Intra-City Integer Overflow Routing (MILP)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from dengue.phase2_config import (
    BED_SURGE_THRESHOLD,
    CITY_COLS,
    TRANSFER_COLS,
)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two (lat, lon) points."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(min(1.0, math.sqrt(max(0.0, a))))


def _extract_ward_coords(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Resolve ward coordinates from Phase-1 WARD_COORDS / attach_coordinates with safe fallback."""
    coords: Dict[str, Tuple[float, float]] = {}

    # 1. Check if coordinates already exist on df
    lat_col = next((c for c in ("lat", "latitude") if c in df.columns), None)
    lon_col = next((c for c in ("lon", "lng", "longitude") if c in df.columns), None)
    if lat_col and lon_col:
        for _, row in df[["ward_id", lat_col, lon_col]].dropna().iterrows():
            coords[str(row["ward_id"])] = (float(row[lat_col]), float(row[lon_col]))

    # 2. Try WARD_COORDS dict from Phase-1 config or ops
    if len(coords) < len(df):
        for mod_name in ("dengue.config", "dengue.ops"):
            try:
                mod = __import__(mod_name, fromlist=["WARD_COORDS"])
                wc = getattr(mod, "WARD_COORDS", None)
                if isinstance(wc, dict):
                    for wid, val in wc.items():
                        wid_s = str(wid)
                        if wid_s in coords:
                            continue
                        if isinstance(val, (tuple, list)) and len(val) >= 2:
                            coords[wid_s] = (float(val[0]), float(val[1]))
                        elif isinstance(val, dict):
                            la = val.get("lat", val.get("latitude"))
                            lo = val.get("lon", val.get("lng", val.get("longitude")))
                            if la is not None and lo is not None:
                                coords[wid_s] = (float(la), float(lo))
            except Exception:
                continue

    # 3. Try Phase-1 dengue.ops.attach_coordinates
    if len(coords) < len(df):
        try:
            from dengue.ops import attach_coordinates

            mini = df[["ward_id", "city"]].drop_duplicates().copy()
            attached = attach_coordinates(mini)
            la_c = next((c for c in ("lat", "latitude") if c in attached.columns), None)
            lo_c = next((c for c in ("lon", "lng", "longitude") if c in attached.columns), None)
            if la_c and lo_c:
                for _, row in attached[["ward_id", la_c, lo_c]].dropna().iterrows():
                    wid_s = str(row["ward_id"])
                    if wid_s not in coords:
                        coords[wid_s] = (float(row[la_c]), float(row[lo_c]))
        except Exception:
            pass

    # 4. Deterministic city-grid fallback for any still-missing ward_id
    city_centers = {
        "Dhaka": (23.8103, 90.4125),
        "Chattogram": (22.3569, 91.7832),
        "Khulna": (22.8456, 89.5403),
        "Rajshahi": (24.3745, 88.6042),
        "Sylhet": (24.8949, 91.8687),
    }
    for idx, row in df[["ward_id", "city"]].drop_duplicates().sort_values("ward_id").iterrows():
        wid_s = str(row["ward_id"])
        if wid_s not in coords:
            base_lat, base_lon = city_centers.get(str(row["city"]), (23.8103, 90.4125))
            offset = (idx % 15) * 0.012
            coords[wid_s] = (base_lat + offset, base_lon + ((idx // 3) % 5) * 0.012)

    return coords


def solve_city_overflow_milp(
    overflow_wards: List[str],
    overflow_amounts: np.ndarray,
    receiver_wards: List[str],
    receiver_headrooms: np.ndarray,
    dist_matrix: np.ndarray,
) -> np.ndarray:
    """Solve exact integer minimum-distance maximum-flow routing via scipy.optimize.milp.

    Returns integer transfer matrix X of shape (len(overflow_wards), len(receiver_wards)).
    """
    n_o = len(overflow_wards)
    n_r = len(receiver_wards)
    x_mat = np.zeros((n_o, n_r), dtype=int)

    if n_o == 0 or n_r == 0:
        return x_mat
    if int(overflow_amounts.sum()) <= 0 or int(receiver_headrooms.sum()) <= 0:
        return x_mat

    k_vars = n_o * n_r
    # Large negative reward per moved patient ensures maximum total patients moved first,
    # and minimum total patient-km distance among all maximum-flow integer solutions.
    max_dist = float(np.max(dist_matrix)) if dist_matrix.size > 0 else 0.0
    move_reward = max(1000.0, max_dist * 10.0 + 1000.0)

    c = np.zeros(k_vars, dtype="float64")
    ub = np.zeros(k_vars, dtype="float64")
    for i in range(n_o):
        for j in range(n_r):
            idx = i * n_r + j
            c[idx] = float(dist_matrix[i, j]) - move_reward
            ub[idx] = float(min(int(overflow_amounts[i]), int(receiver_headrooms[j])))

    # Constraints:
    #   Row 0..n_o-1:      sum_j x[i, j] <= overflow_amounts[i]
    #   Row n_o..n_o+n_r-1: sum_i x[i, j] <= receiver_headrooms[j]
    a_mat = np.zeros((n_o + n_r, k_vars), dtype="float64")
    for i in range(n_o):
        a_mat[i, i * n_r : (i + 1) * n_r] = 1.0
    for j in range(n_r):
        for i in range(n_o):
            a_mat[n_o + j, i * n_r + j] = 1.0

    lb_c = np.zeros(n_o + n_r, dtype="float64")
    ub_c = np.concatenate(
        [overflow_amounts.astype("float64"), receiver_headrooms.astype("float64")]
    )

    result = milp(
        c=c,
        integrality=np.ones(k_vars, dtype=int),
        bounds=Bounds(lb=np.zeros(k_vars, dtype="float64"), ub=ub),
        constraints=LinearConstraint(a_mat, lb_c, ub_c),
    )

    if result.success and result.x is not None:
        sol = np.maximum(0, np.rint(result.x)).astype(int)
        return sol.reshape((n_o, n_r))

    # Deterministic greedy nearest-receiver integer fallback if MILP ever fails to converge
    rem_o = overflow_amounts.astype(int).copy()
    rem_r = receiver_headrooms.astype(int).copy()
    pairs = [
        (float(dist_matrix[i, j]), overflow_wards[i], receiver_wards[j], i, j)
        for i in range(n_o)
        for j in range(n_r)
    ]
    pairs.sort()
    for _, _, _, i, j in pairs:
        if rem_o[i] > 0 and rem_r[j] > 0:
            moved = min(int(rem_o[i]), int(rem_r[j]))
            x_mat[i, j] = moved
            rem_o[i] -= moved
            rem_r[j] -= moved
    return x_mat


def route_hospital_overflow(
    wards: pd.DataFrame,
    params: Any,
    coords_override: Dict[str, Tuple[float, float]] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """Run R2 hospital surge bands, intra-city MILP overflow routing, and city rollups."""
    df = wards.copy().reset_index(drop=True)
    reserve_beds = max(0, int(getattr(params, "reserve_beds", 0)))

    # 1. Pre-routing surge bands (Bagust et al., BMJ 1999: <85% Normal, 85-100% Surge, >100% Overflow)
    req = df["beds_required"].astype(int).to_numpy()
    cap = df["bed_capacity"].astype(int).to_numpy()
    util = np.where(cap > 0, req / np.maximum(cap, 1), 0.0)

    df["bed_utilization_pct"] = np.round(100.0 * util, 1)
    df["bed_status"] = np.select(
        [req > cap, util >= BED_SURGE_THRESHOLD - 1e-9],
        ["Overflow", "Surge Warning"],
        default="Normal",
    )
    df["bed_overflow"] = np.maximum(0, req - cap).astype(int)

    routed_out = np.zeros(len(df), dtype=int)
    routed_in = np.zeros(len(df), dtype=int)

    coords = coords_override if coords_override is not None else _extract_ward_coords(df)
    transfer_records: List[Dict[str, Any]] = []

    # 2. Solve integer MILP independently within each city (strictly intra-city)
    for city_name, grp in df.groupby("city", sort=True):
        grp_sorted = grp.sort_values("ward_id", kind="mergesort")
        ov_rows = grp_sorted[grp_sorted["bed_overflow"] > 0]
        if ov_rows.empty:
            continue

        # Receiver available capacity respects reserve_beds (applied ONLY to receivers)
        rec_candidates = grp_sorted[grp_sorted["bed_overflow"] == 0].copy()
        rec_headroom_series = (
            rec_candidates["bed_capacity"].astype(int)
            - rec_candidates["beds_required"].astype(int)
            - reserve_beds
        ).clip(lower=0)
        rec_candidates = rec_candidates[rec_headroom_series > 0]
        rec_headrooms = rec_headroom_series.loc[rec_candidates.index].to_numpy(dtype=int)

        if rec_candidates.empty:
            continue

        ov_indices = ov_rows.index.to_numpy()
        rec_indices = rec_candidates.index.to_numpy()
        ov_wards = ov_rows["ward_id"].astype(str).tolist()
        rec_wards = rec_candidates["ward_id"].astype(str).tolist()
        ov_amounts = ov_rows["bed_overflow"].astype(int).to_numpy()

        dist_mat = np.zeros((len(ov_wards), len(rec_wards)), dtype="float64")
        for i, fw in enumerate(ov_wards):
            la1, lo1 = coords[fw]
            for j, tw in enumerate(rec_wards):
                la2, lo2 = coords[tw]
                dist_mat[i, j] = round(_haversine_km(la1, lo1, la2, lo2), 3)

        x_mat = solve_city_overflow_milp(
            overflow_wards=ov_wards,
            overflow_amounts=ov_amounts,
            receiver_wards=rec_wards,
            receiver_headrooms=rec_headrooms,
            dist_matrix=dist_mat,
        )

        for i, ov_idx in enumerate(ov_indices):
            for j, rec_idx in enumerate(rec_indices):
                moved = int(x_mat[i, j])
                if moved > 0:
                    routed_out[ov_idx] += moved
                    routed_in[rec_idx] += moved
                    transfer_records.append(
                        {
                            "city": str(city_name),
                            "from_ward": ov_wards[i],
                            "to_ward": rec_wards[j],
                            "patients_moved": moved,
                            "distance_km": round(float(dist_mat[i, j]), 2),
                        }
                    )

    df["beds_routed_out"] = routed_out
    df["beds_routed_in"] = routed_in
    df["bed_net_deficit"] = np.maximum(0, df["bed_overflow"].to_numpy(dtype=int) - routed_out).astype(int)

    # 3. Recompute operational alert flags using post-routing bed_net_deficit
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

    # 4. Deterministic transfers table
    if transfer_records:
        transfers = (
            pd.DataFrame(transfer_records, columns=list(TRANSFER_COLS))
            .sort_values(["city", "from_ward", "to_ward"], kind="mergesort")
            .reset_index(drop=True)
        )
    else:
        transfers = pd.DataFrame(columns=list(TRANSFER_COLS))

    # 5. City-level utilization, remaining headroom, and escalation status
    city_rows: List[Dict[str, Any]] = []
    for city_name, grp in df.groupby("city", sort=True):
        beds_req = int(grp["beds_required"].sum())
        bed_cap = int(grp["bed_capacity"].sum())
        beds_routed = int(grp["beds_routed_out"].sum())
        beds_unmet = int(grp["bed_net_deficit"].sum())
        # Remaining free beds across the city after receiving routed overflow
        post_load = grp["beds_required"].to_numpy(dtype=int) + grp["beds_routed_in"].to_numpy(dtype=int)
        remaining_headroom = int(np.maximum(0, grp["bed_capacity"].to_numpy(dtype=int) - post_load).sum())
        exhausted = beds_unmet > 0

        city_rows.append(
            {
                "city": str(city_name),
                "total_wards": int(len(grp)),
                "beds_required": beds_req,
                "bed_capacity": bed_cap,
                "beds_routed": beds_routed,
                "beds_unmet": beds_unmet,
                "utilization_pct": round(100.0 * beds_req / bed_cap, 1) if bed_cap > 0 else 0.0,
                "remaining_headroom": remaining_headroom,
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

    # 6. Escalation alerts (CODE BLACK & city pool exhausted)
    alerts: List[Dict[str, Any]] = []
    for _, row in df[df["code_black"]].sort_values("ward_id").iterrows():
        alerts.append(
            {
                "level": "CODE_BLACK",
                "city": str(row["city"]),
                "ward_id": str(row["ward_id"]),
                "message": (
                    f"CODE BLACK in {row['ward_id']} ({row['city']}): "
                    f"unmet bed deficit {int(row['bed_net_deficit'])}, "
                    f"saline deficit {int(row['saline_deficit'])}."
                ),
            }
        )
    for _, crow in cities[cities["city_pool_exhausted"]].sort_values("city").iterrows():
        alerts.append(
            {
                "level": "CITY_EXHAUSTED",
                "city": str(crow["city"]),
                "ward_id": None,
                "message": (
                    f"city pool exhausted - escalate ({crow['city']}): "
                    f"{int(crow['beds_unmet'])} unmet bed(s) after routing {int(crow['beds_routed'])}."
                ),
            }
        )

    df = df.sort_values("ward_id", kind="mergesort").reset_index(drop=True)
    return df, transfers, cities, alerts