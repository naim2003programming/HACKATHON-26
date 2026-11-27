"""R5 Offline resilience, deterministic forecast ladder, and precision lock."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from dengue.phase2_config import (
    CEIL_EPSILON,
    DATA_DIR,
    FORECAST_DECIMALS,
    MODELS_DIR,
    RAW_DATA_DIR,
    ROOT_DIR,
)

_CONN_CACHE: dict[str, float | bool] = {"ts": 0.0, "online": False}
_CONN_TTL_SECONDS = 60.0


def safe_ceil(values) -> np.ndarray:
    """Ceiling with epsilon guard against float32/float64 rounding drift."""
    arr = np.asarray(values, dtype="float64")
    return np.maximum(0, np.ceil(arr - CEIL_EPSILON)).astype(int)


def is_offline_forced(offline: bool = False) -> bool:
    """Return True when offline mode is requested via flag or environment."""
    return bool(offline) or os.environ.get("ALERTAHEAD_OFFLINE", "").strip() == "1"


def check_connectivity(offline: bool = False, timeout: float = 1.5) -> bool:
    """Lazy, 60s-cached connectivity check. Zero network calls when offline."""
    if is_offline_forced(offline):
        return False

    now = time.monotonic()
    if _CONN_CACHE["ts"] > 0 and (now - float(_CONN_CACHE["ts"])) < _CONN_TTL_SECONDS:
        return bool(_CONN_CACHE["online"])

    online = False
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=timeout):
            online = True
    except OSError:
        online = False

    _CONN_CACHE["ts"] = now
    _CONN_CACHE["online"] = online
    return online


def reset_connectivity_cache() -> None:
    """Clear connectivity cache (used in tests)."""
    _CONN_CACHE["ts"] = 0.0
    _CONN_CACHE["online"] = False


def get_git_short_hash() -> str:
    """Return local git short commit hash without any network access."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT_DIR),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def find_predictions_parquet() -> Optional[Path]:
    """Locate predictions.parquet in standard repository directories."""
    candidates = [
        DATA_DIR / "predictions.parquet",
        MODELS_DIR / "predictions.parquet",
        ROOT_DIR / "predictions.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    found = sorted(ROOT_DIR.glob("**/predictions.parquet"))
    return found[0] if found else None


def load_predictions_parquet() -> Optional[pd.DataFrame]:
    """Load precomputed forecasts parquet if present."""
    path = find_predictions_parquet()
    if path is None:
        return None
    df = pd.read_parquet(path)
    date_col = "date" if "date" in df.columns else next(
        (c for c in ("week_start", "week_date", "ds") if c in df.columns), None
    )
    if date_col and date_col != "date":
        df = df.rename(columns={date_col: "date"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "ward_id" not in df.columns and "ward" in df.columns:
        df = df.rename(columns={"ward": "ward_id"})
    return df


def load_ward_metadata() -> pd.DataFrame:
    """Load static/latest ward traits (density, cases_now, city) from local dataset."""
    candidates = []
    for folder in (RAW_DATA_DIR, DATA_DIR):
        if folder.exists():
            candidates.extend(sorted(folder.glob("*.csv")))
            candidates.extend(sorted(folder.glob("*.xlsx")))

    for path in candidates:
        try:
            df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
        except Exception:
            continue
        ward_col = "ward_id" if "ward_id" in df.columns else ("ward" if "ward" in df.columns else None)
        if ward_col is None:
            continue
        date_col = next((c for c in ("date", "week_start", "week_date", "ds") if c in df.columns), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values([ward_col, date_col])
        out = pd.DataFrame()
        out["ward_id"] = df[ward_col].astype(str)
        if "city" in df.columns:
            out["city"] = df["city"].astype(str)
        if date_col:
            out["date"] = df[date_col]
        if "population_density_sqkm" in df.columns:
            out["population_density_sqkm"] = df["population_density_sqkm"].astype("float64")
        cases_col = next(
            (c for c in ("cases", "dengue_cases", "weekly_cases", "reported_cases") if c in df.columns),
            None,
        )
        if cases_col:
            out["cases_actual"] = df[cases_col].astype("float64")
        return out

    return pd.DataFrame(columns=["ward_id", "city", "date", "population_density_sqkm", "cases_actual"])


def _normalize_forecast_frame(
    df: pd.DataFrame,
    h: int,
    source: str,
) -> Tuple[pd.DataFrame, str]:
    """Enforce float64 + round(3) and deterministic ward_id ordering."""
    out = df.copy()
    pred_col = f"pred_h{h}"
    high_col = f"high_h{h}"

    if "ward_id" not in out.columns and "ward" in out.columns:
        out = out.rename(columns={"ward": "ward_id"})
    out["ward_id"] = out["ward_id"].astype(str)
    if "city" not in out.columns:
        out["city"] = "Unknown"
    out["city"] = out["city"].astype(str)

    if high_col not in out.columns:
        out[high_col] = out[pred_col]

    for col in (pred_col, high_col):
        out[col] = out[col].astype("float64").round(FORECAST_DECIMALS)

    if "pred_h4" in out.columns:
        out["pred_h4"] = out["pred_h4"].astype("float64").round(FORECAST_DECIMALS)

    out = out.sort_values("ward_id", kind="mergesort").reset_index(drop=True)
    return out, source


def _seasonal_last_year_forecast(
    raw_df: pd.DataFrame,
    issue_ts: pd.Timestamp,
    h: int,
) -> pd.DataFrame:
    """Last-resort deterministic fallback: same-week-last-year cases."""
    target_last_year = issue_ts + pd.Timedelta(weeks=h) - pd.Timedelta(weeks=52)
    dates = raw_df["date"].dropna().unique()
    if len(dates) == 0:
        raise ValueError("No dates available for seasonal fallback.")
    nearest = min(dates, key=lambda d: abs(pd.Timestamp(d) - target_last_year))
    sub = raw_df.loc[raw_df["date"] == nearest, ["ward_id", "city", "cases_actual"]].copy()
    sub[f"pred_h{h}"] = sub["cases_actual"].astype("float64").clip(lower=0.0)
    sub[f"high_h{h}"] = (sub[f"pred_h{h}"] * 1.25).round(FORECAST_DECIMALS)
    return sub[["ward_id", "city", f"pred_h{h}", f"high_h{h}"]]


def get_forecast(
    preds: Optional[pd.DataFrame],
    issue_date: str | pd.Timestamp,
    h: int,
    raw_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, str]:
    """Deterministic forecast ladder locked at float64 + round(3).

    Order for allocation/routing/notice:
      1. predictions.parquet (exact issue_date match -> identical online & offline)
      2. Live LightGBM (via Phase-1 engine if models/ available)
      3. Same-week-last-year seasonal fallback
    """
    issue_ts = pd.Timestamp(issue_date)
    pred_col = f"pred_h{h}"
    high_col = f"high_h{h}"

    # 1. Parquet (primary source for consistent online/offline numbers)
    if preds is None:
        preds = load_predictions_parquet()

    if preds is not None and not preds.empty and pred_col in preds.columns:
        p_dates = pd.to_datetime(preds["date"])
        mask = p_dates == issue_ts
        if mask.any():
            keep = ["ward_id", "city", pred_col]
            if high_col in preds.columns:
                keep.append(high_col)
            if "pred_h4" in preds.columns and "pred_h4" not in keep:
                keep.append("pred_h4")
            if "tier" in preds.columns:
                keep.append("tier")
            sub = preds.loc[mask, keep].copy()
            return _normalize_forecast_frame(sub, h, "parquet")

    # 2. Live LightGBM fallback (if parquet unavailable or issue_date outside parquet)
    try:
        from dengue import data as p1_data
        from dengue import engine as p1_engine
        from dengue import features as p1_features
        from dengue import model as p1_model

        if MODELS_DIR.exists() and any(MODELS_DIR.glob("*.joblib")):
            raw = p1_data.load_raw_data()
            feat = p1_features.build_features(raw)
            models = {
                hz: p1_model.load_model(hz)
                for hz in (2, 3, 4)
                if hasattr(p1_model, "load_model")
            }
            if models:
                tbl = p1_engine.predict_table(feat, models)
                tbl["date"] = pd.to_datetime(tbl["date"])
                mask = tbl["date"] == issue_ts
                if mask.any() and pred_col in tbl.columns:
                    keep = [c for c in ("ward_id", "city", pred_col, high_col, "pred_h4") if c in tbl.columns]
                    return _normalize_forecast_frame(tbl.loc[mask, keep], h, "live_lightgbm")
    except Exception:
        pass

    # 3. Same-week-last-year deterministic fallback
    meta = raw_df if raw_df is not None else load_ward_metadata()
    if meta is not None and not meta.empty and "cases_actual" in meta.columns:
        sub = _seasonal_last_year_forecast(meta, issue_ts, h)
        return _normalize_forecast_frame(sub, h, "fallback: seasonal")

    raise RuntimeError(
        f"Unable to produce forecast for issue_date={issue_date}, h={h}: "
        "no parquet, models, or raw dataset found."
    )