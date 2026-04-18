import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


WINDOWS = [20, 50, 100]
GAP = 10
LOCATION_WINDOWS = [200, 500]
MAIN_WINDOW = 50
SUBSAMPLE_STEP = MAIN_WINDOW
MIN_DT_SECONDS = 0.001  # 1ms floor to avoid unstable speed spikes.
SPEED_CAP = 1000.0


def _parse_timestamp_int(value) -> Optional[int]:
    """Return validated integer timestamp (ms), else None."""
    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value if value > 0 else None

    if isinstance(value, float):
        if not np.isfinite(value) or not value.is_integer():
            return None
        iv = int(value)
        return iv if iv > 0 else None

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.isdigit():
            iv = int(s)
            return iv if iv > 0 else None
        try:
            f = float(s)
        except ValueError:
            return None
        if not np.isfinite(f) or not f.is_integer():
            return None
        iv = int(f)
        return iv if iv > 0 else None

    return None


def _parse_line(line: str) -> Optional[Tuple[int, float, float, str, float, bool]]:
    """
    Parse one JSONL line into a normalized trade tuple:
    (timestamp_ms, price, quantity, side, delta, reconstructed)

    Supported schemas:
    1) Raw: {timestamp, price, quantity, side}
    2) Wrapped: {ts, flow:{timestamp, price, delta,...}}
       quantity reconstructed as abs(delta), side from sign(delta).
    """
    try:
        obj = json.loads(line)
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None

    # Schema 1: raw trade object
    if all(k in obj for k in ("timestamp", "price", "quantity", "side")):
        try:
            ts = _parse_timestamp_int(obj["timestamp"])
            price = float(obj["price"])
            qty = float(obj["quantity"])
            side = str(obj["side"]).lower()
        except (TypeError, ValueError):
            return None

        if ts is None or price <= 0 or qty <= 0 or side not in {"buy", "sell"}:
            return None

        delta = qty if side == "buy" else -qty

        return ts, price, qty, side, delta, False

    # Schema 2: wrapped/enriched object with flow payload
    flow = obj.get("flow")
    if isinstance(flow, dict) and "price" in flow and "delta" in flow and "ts" in obj:
        try:
            # Requirement: wrapped schema timestamp must come from top-level ts.
            ts = _parse_timestamp_int(obj["ts"])
            price = float(flow["price"])
            delta = float(flow["delta"])
        except (TypeError, ValueError):
            return None

        if ts is None or price <= 0:
            return None

        # Cannot infer side from zero-delta rows reliably.
        if delta == 0:
            return None

        qty = abs(delta)
        side = "buy" if delta > 0 else "sell"

        if qty <= 0:
            return None

        return ts, price, qty, side, delta, True

    return None


def load_trades(input_path: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    rows: List[Tuple[int, float, float, str, float, bool]] = []
    stats = {
        "lines_total": 0,
        "rows_parsed": 0,
        "rows_bad": 0,
        "rows_raw": 0,
        "rows_reconstructed": 0,
    }

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            stats["lines_total"] += 1
            s = line.strip()
            if not s:
                stats["rows_bad"] += 1
                continue

            parsed = _parse_line(s)
            if parsed is None:
                stats["rows_bad"] += 1
                continue

            rows.append(parsed)
            stats["rows_parsed"] += 1
            if parsed[-1]:
                stats["rows_reconstructed"] += 1
            else:
                stats["rows_raw"] += 1

    if not rows:
        raise ValueError("No valid trade rows parsed from input file.")

    df = pd.DataFrame(
        rows,
        columns=["timestamp", "price", "quantity", "side", "delta", "reconstructed"],
    )

    # Ensure ascending time order using stable sort.
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    return df, stats


def _causal_tertile_bins(
    series: pd.Series, labels: List[str], min_history: int = 100
) -> pd.Series:
    """
    Causal tertile bins with strict no-lookahead behavior.

    Bin boundaries at row t are computed only from rows < t via shift(1).
    """
    hist = series.shift(1)
    q1 = hist.expanding(min_periods=min_history).quantile(1 / 3)
    q2 = hist.expanding(min_periods=min_history).quantile(2 / 3)

    out = pd.Series(index=series.index, dtype="object")

    low_mask = series <= q1
    mid_mask = (series > q1) & (series <= q2)
    high_mask = series > q2

    out.loc[low_mask] = labels[0]
    out.loc[mid_mask] = labels[1]
    out.loc[high_mask] = labels[2]
    return out


def _direction_from_target(target: pd.Series) -> pd.Series:
    """
    Encode direction causally without forcing flat/NaN values into "down".

    Returns:
      1 for target > 0
      0 for target < 0
      NaN for target == 0 or missing target
    """
    out = pd.Series(np.nan, index=target.index, dtype="float64")
    out.loc[target > 0] = 1.0
    out.loc[target < 0] = 0.0
    return out


def build_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["event_idx"] = np.arange(len(out), dtype="int64")

    # Keep normalized delta from parser (raw or reconstructed source).

    for n in WINDOWS:
        out[f"delta_{n}"] = out["delta"].rolling(window=n, min_periods=n).sum()
        out[f"volume_{n}"] = out["quantity"].rolling(window=n, min_periods=n).sum()
        out[f"pressure_{n}"] = out[f"delta_{n}"] / out[f"volume_{n}"].replace(0, np.nan)
        out[f"range_{n}"] = (
            out["price"].rolling(window=n, min_periods=n).max()
            - out["price"].rolling(window=n, min_periods=n).min()
        )

        dt_seconds = (out["timestamp"] - out["timestamp"].shift(n)) / 1000.0
        raw_speed = np.where(dt_seconds >= MIN_DT_SECONDS, n / dt_seconds, np.nan)
        out[f"speed_{n}"] = np.clip(raw_speed, None, SPEED_CAP)

        out[f"target_{n}"] = out["price"].shift(-n) - out["price"]
        out[f"target_direction_{n}"] = _direction_from_target(out[f"target_{n}"])

        # Gap-based target to reduce immediate microstructure persistence effects.
        out[f"target_{n}_gap"] = out["price"].shift(-(GAP + n)) - out["price"].shift(-GAP)
        out[f"target_direction_{n}_gap"] = _direction_from_target(out[f"target_{n}_gap"])

    # Location features from local rolling structure.
    shifted_price = out["price"].shift(1)
    for n in LOCATION_WINDOWS:
        out[f"rolling_high_{n}"] = shifted_price.rolling(window=n, min_periods=n).max()
        out[f"rolling_low_{n}"] = shifted_price.rolling(window=n, min_periods=n).min()
        out[f"rolling_mean_{n}"] = shifted_price.rolling(window=n, min_periods=n).mean()

        local_range = out[f"rolling_high_{n}"] - out[f"rolling_low_{n}"]
        out[f"position_{n}"] = (out["price"] - out[f"rolling_low_{n}"]) / local_range.replace(
            0, np.nan
        )

    # Primary location bin from position_200.
    out["location_bin"] = np.select(
        [out["position_200"] <= 0.2, out["position_200"] >= 0.8],
        ["low", "high"],
        default="mid",
    )
    out.loc[out["position_200"].isna(), "location_bin"] = np.nan

    # Absorption flag (N=50)
    # Causal thresholds: percentile boundaries use strictly prior rows only.
    out["vol_p80_50"] = out["volume_50"].shift(1).expanding(min_periods=50).quantile(0.80)
    out["range_p30_50"] = out["range_50"].shift(1).expanding(min_periods=50).quantile(0.30)
    out["absorption_flag"] = (
        (out["volume_50"] > out["vol_p80_50"]) & (out["range_50"] < out["range_p30_50"])
    ).astype("int64")

    # Discretization columns (using N=50 representative values), causal-only.
    out["pressure_bin"] = _causal_tertile_bins(
        out["pressure_50"], ["low", "mid", "high"], min_history=100
    )
    out["speed_bin"] = _causal_tertile_bins(
        out["speed_50"], ["slow", "medium", "fast"], min_history=100
    )
    out["range_bin"] = _causal_tertile_bins(
        out["range_50"], ["low", "medium", "high"], min_history=100
    )

    # Drop flat preferred target rows explicitly to avoid directional label bias.
    out = out[out["target_50_gap"] != 0].copy()

    required_columns = [
        "pressure_20",
        "pressure_50",
        "pressure_100",
        "delta_20",
        "delta_50",
        "delta_100",
        "range_20",
        "range_50",
        "range_100",
        "speed_20",
        "speed_50",
        "speed_100",
        "absorption_flag",
        "target_20",
        "target_50",
        "target_100",
        "target_direction_20",
        "target_direction_50",
        "target_direction_100",
        "target_20_gap",
        "target_50_gap",
        "target_100_gap",
        "target_direction_20_gap",
        "target_direction_50_gap",
        "target_direction_100_gap",
        "pressure_bin",
        "speed_bin",
        "range_bin",
        "rolling_high_200",
        "rolling_low_200",
        "rolling_mean_200",
        "position_200",
        "rolling_high_500",
        "rolling_low_500",
        "rolling_mean_500",
        "position_500",
        "location_bin",
    ]

    out = out.dropna(subset=required_columns).copy()

    # Direction columns are guaranteed non-null post-filter; cast to int.
    for n in WINDOWS:
        out[f"target_direction_{n}"] = out[f"target_direction_{n}"].astype("int64")
        out[f"target_direction_{n}_gap"] = out[f"target_direction_{n}_gap"].astype("int64")

    # Keep only final output schema.
    final_columns = [
        "timestamp",
        "price",
        "quantity",
        "side",
        "delta",
        "reconstructed",
        "pressure_20",
        "pressure_50",
        "pressure_100",
        "delta_20",
        "delta_50",
        "delta_100",
        "range_20",
        "range_50",
        "range_100",
        "speed_20",
        "speed_50",
        "speed_100",
        "absorption_flag",
        "target_20",
        "target_50",
        "target_100",
        "target_direction_20",
        "target_direction_50",
        "target_direction_100",
        "target_20_gap",
        "target_50_gap",
        "target_100_gap",
        "target_direction_20_gap",
        "target_direction_50_gap",
        "target_direction_100_gap",
        "pressure_bin",
        "speed_bin",
        "range_bin",
        "rolling_high_200",
        "rolling_low_200",
        "rolling_mean_200",
        "position_200",
        "rolling_high_500",
        "rolling_low_500",
        "rolling_mean_500",
        "position_500",
        "location_bin",
    ]
    return out[final_columns].reset_index(drop=True)


def print_validation(df_out: pd.DataFrame, parse_stats: Dict[str, int]) -> None:
    core_cols = [
        "pressure_20",
        "pressure_50",
        "pressure_100",
        "delta_20",
        "delta_50",
        "delta_100",
        "range_20",
        "range_50",
        "range_100",
        "speed_20",
        "speed_50",
        "speed_100",
        "target_20",
        "target_50",
        "target_100",
    ]

    print("=== Build Summary ===")
    print(f"lines_total         : {parse_stats['lines_total']}")
    print(f"rows_parsed_valid   : {parse_stats['rows_parsed']}")
    print(f"rows_bad_skipped    : {parse_stats['rows_bad']}")
    print(f"rows_raw_used       : {parse_stats['rows_raw']}")
    print(f"rows_reconstructed  : {parse_stats['rows_reconstructed']}")
    print(f"rows_before_dedup   : {parse_stats['rows_parsed']}")
    print("rows_dedup_dropped  : 0")
    print(f"rows_after_dedup    : {parse_stats['rows_parsed']}")
    print(
        f"rows_before_subsample: {parse_stats.get('rows_before_subsample', 0)}"
    )
    print(
        f"rows_after_subsample : {parse_stats.get('rows_after_subsample', 0)}"
    )
    print(
        f"pct_removed_subsample: {parse_stats.get('pct_removed_subsample', 0.0):.2f}%"
    )
    print(f"rows_output         : {len(df_out)}")

    inf_count = int(np.isinf(df_out[core_cols].to_numpy()).sum())
    nan_count = int(df_out[core_cols].isna().sum().sum())

    print("\n=== Validation ===")
    print(f"core_nan_count      : {nan_count}")
    print(f"core_inf_count      : {inf_count}")

    print("\n=== Distribution Snapshots ===")
    for col in ["pressure_50", "speed_50", "range_50", "target_50"]:
        s = df_out[col]
        print(
            f"{col:<16} min={s.min():.8f} median={s.median():.8f} max={s.max():.8f}"
        )

    s50 = df_out["speed_50"]
    print(
        "speed_50_summary   "
        f"min={s50.min():.8f} p50={s50.quantile(0.5):.8f} "
        f"p99={s50.quantile(0.99):.8f} max={s50.max():.8f}"
    )

    print("\nBin counts:")
    for c in ["pressure_bin", "speed_bin", "range_bin"]:
        counts = df_out[c].value_counts().to_dict()
        print(f"  {c}: {counts}")
    print(f"  location_bin: {df_out['location_bin'].value_counts().to_dict()}")

    print("\nNotes:")
    print("- Deduplication disabled: all parsed events are retained.")
    print(
        "- No lookahead in location features: rolling high/low/mean use price.shift(1)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build event-based feature dataset from trade JSONL file."
    )
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument(
        "--output",
        default="event_features_dataset.csv",
        help="Output CSV path (default: event_features_dataset.csv)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df_raw, parse_stats = load_trades(input_path)
    df_full = build_feature_dataset(df_raw)

    rows_before_subsample = len(df_full)
    df_out = df_full.iloc[::SUBSAMPLE_STEP].copy().reset_index(drop=True)
    rows_after_subsample = len(df_out)

    parse_stats["rows_before_subsample"] = rows_before_subsample
    parse_stats["rows_after_subsample"] = rows_after_subsample
    parse_stats["pct_removed_subsample"] = (
        100.0 * (rows_before_subsample - rows_after_subsample) / rows_before_subsample
        if rows_before_subsample > 0
        else 0.0
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)

    print_validation(df_out, parse_stats)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
