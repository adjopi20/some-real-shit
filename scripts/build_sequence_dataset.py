import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "pressure_50",
    "pressure_bin",
    "speed_50",
    "speed_bin",
    "absorption_flag",
    "location_bin",
    "regime",
    "direction",
    "price",
    "target_50_gap",
]


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype="float64")
    x_centered = x - x.mean()
    denom = float((x_centered**2).sum())

    def slope(arr: np.ndarray) -> float:
        y = arr.astype("float64")
        y_centered = y - y.mean()
        return float((x_centered * y_centered).sum() / denom)

    return series.rolling(window=window, min_periods=window).apply(slope, raw=True)


def _make_sequence_string(series: pd.Series, window: int) -> pd.Series:
    seq_df = pd.concat([series.shift(i) for i in range(window - 1, -1, -1)], axis=1)

    def build(row: pd.Series):
        if row.isna().any():
            return np.nan
        return ",".join(row.astype(str).tolist())

    return seq_df.apply(build, axis=1)


def _classify_absorption_pattern(seq_str: str) -> str | float:
    if not isinstance(seq_str, str):
        return np.nan

    vals = seq_str.split(",")
    arr = np.array([1 if v == "present" else 0 for v in vals], dtype="int64")

    if arr.sum() == 0:
        return "none"

    if arr[-1] == 1:
        trailing_ones = 0
        for v in arr[::-1]:
            if v == 1:
                trailing_ones += 1
            else:
                break
        if trailing_ones >= 2:
            return "persists"
        return "appears"

    return "fades"


def _future_state_counts(state_pressure: pd.Series, forward: int):
    future = pd.concat([state_pressure.shift(-k) for k in range(1, forward + 1)], axis=1)
    future.columns = [f"t+{k}" for k in range(1, forward + 1)]
    return future


def build_sequence_dataset(df: pd.DataFrame, lookback: int, forward: int) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()

    # --- Step 1: state columns (using existing feature outputs only)
    out["state_pressure"] = np.select(
        [out["pressure_bin"].eq("high"), out["pressure_bin"].eq("low")],
        ["strong_up", "strong_down"],
        default="neutral",
    )
    out["state_speed"] = out["speed_bin"]
    out["state_absorption"] = np.where(out["absorption_flag"].eq(1), "present", "absent")
    out["state_location"] = out["location_bin"]
    out["state_regime"] = out["regime"]
    out["state_key"] = (
        out["state_regime"].astype(str)
        + "|"
        + out["state_pressure"].astype(str)
        + "|"
        + out["state_location"].astype(str)
        + "|"
        + out["state_absorption"].astype(str)
    )

    # --- Step 2: sequence features (causal lookback only)
    out[f"pressure_seq_{lookback}"] = _make_sequence_string(out["state_pressure"], lookback)
    out[f"speed_seq_{lookback}"] = _make_sequence_string(out["state_speed"], lookback)
    out[f"absorption_seq_{lookback}"] = _make_sequence_string(
        out["state_absorption"], lookback
    )

    pressure_slope = _rolling_slope(out["pressure_50"], lookback)
    out[f"pressure_trend_{lookback}"] = np.select(
        [pressure_slope > 0.05, pressure_slope < -0.05],
        ["increasing", "decreasing"],
        default="flat",
    )

    speed_slope = _rolling_slope(out["speed_50"], lookback)
    speed_mean = out["speed_50"].rolling(window=lookback, min_periods=lookback).mean()
    relative_speed_slope = speed_slope / speed_mean.replace(0, np.nan)
    out[f"speed_trend_{lookback}"] = np.select(
        [relative_speed_slope > 0.05, relative_speed_slope < -0.05],
        ["accelerating", "decelerating"],
        default="stable",
    )

    out[f"absorption_pattern_{lookback}"] = out[f"absorption_seq_{lookback}"].apply(
        _classify_absorption_pattern
    )

    # --- Step 3/4: resolution detection + sequence labels (uses forward rows only for labels)
    dir_sign = np.select(
        [out["state_pressure"].eq("strong_up"), out["state_pressure"].eq("strong_down")],
        [1, -1],
        default=0,
    ).astype("int64")
    out["direction_sign"] = dir_sign

    future_pressure = _future_state_counts(out["state_pressure"], forward)
    same_side_count = pd.Series(0, index=out.index, dtype="int64")
    opposite_side_count = pd.Series(0, index=out.index, dtype="int64")

    for col in future_pressure.columns:
        same_side_count += (future_pressure[col] == out["state_pressure"]).astype("int64")
        opposite_side_count += (
            ((future_pressure[col] == "strong_up") & (out["state_pressure"] == "strong_down"))
            | ((future_pressure[col] == "strong_down") & (out["state_pressure"] == "strong_up"))
        ).astype("int64")

    future_absorption_present = pd.Series(False, index=out.index)
    for k in range(1, forward + 1):
        future_absorption_present = future_absorption_present | out["absorption_flag"].shift(
            -k
        ).eq(1)

    price_move_forward = out["price"].shift(-forward) - out["price"]
    out[f"price_move_t_plus_{forward}"] = price_move_forward
    out["future_outcome_sign"] = np.sign(out["target_50_gap"]).astype("int64")

    strong_pressure = out["direction_sign"] != 0
    aligned_price_move = ((out["direction_sign"] == 1) & (price_move_forward > 0)) | (
        (out["direction_sign"] == -1) & (price_move_forward < 0)
    )
    opposite_price_move = ((out["direction_sign"] == 1) & (price_move_forward < 0)) | (
        (out["direction_sign"] == -1) & (price_move_forward > 0)
    )

    continuation = strong_pressure & (same_side_count >= 3) & aligned_price_move
    failure = strong_pressure & (
        ((same_side_count <= 1) & (~aligned_price_move))
        | (future_absorption_present & (opposite_side_count >= 1))
    )
    reversal = (
        strong_pressure
        & (out["state_absorption"].eq("present") | future_absorption_present)
        & (opposite_side_count >= 2)
        & opposite_price_move
    )

    out["context_valid"] = (
        out["state_regime"].eq("trend")
        & out["state_location"].isin(["low", "high"])
        & strong_pressure
    ).astype("boolean")

    out["resolution_label"] = "no_resolution"
    out.loc[(out["state_regime"].eq("trend") & continuation & ~out["context_valid"]), "resolution_label"] = "continuation"
    out.loc[out["context_valid"] & failure, "resolution_label"] = "breakout_failure"
    out.loc[out["context_valid"] & continuation, "resolution_label"] = "breakout_success"
    out.loc[reversal, "resolution_label"] = "reversal"

    # Rows lacking complete history/forward horizon should not have sequence resolution labels.
    insufficient_history = pd.Series(False, index=out.index)
    insufficient_forward = pd.Series(False, index=out.index)
    insufficient_history.iloc[: lookback - 1] = True
    insufficient_forward.iloc[len(out) - forward :] = True

    seq_cols = [
        f"pressure_seq_{lookback}",
        f"speed_seq_{lookback}",
        f"absorption_seq_{lookback}",
        f"pressure_trend_{lookback}",
        f"speed_trend_{lookback}",
        f"absorption_pattern_{lookback}",
    ]
    out.loc[insufficient_history, seq_cols] = np.nan
    out.loc[insufficient_forward, "resolution_label"] = np.nan
    out.loc[insufficient_forward, "context_valid"] = pd.NA

    winner_side = np.where(
        out["resolution_label"].isin(["breakout_success", "continuation"]),
        out["direction_sign"],
        np.where(
            out["resolution_label"].isin(["breakout_failure", "reversal"]),
            -out["direction_sign"],
            0,
        ),
    )
    out["winner_side"] = winner_side.astype("int64")
    out["winner_confirmed"] = (
        (out["winner_side"] != 0) & (out["winner_side"] == out["future_outcome_sign"])
    ).astype("int64")

    return out


def build_sequence_pattern_summary(df_seq: pd.DataFrame, lookback: int) -> pd.DataFrame:
    group_cols = ["state_key", f"pressure_trend_{lookback}", f"absorption_pattern_{lookback}"]
    use = df_seq.dropna(subset=group_cols + ["resolution_label"]).copy()

    grouped = use.groupby(group_cols, dropna=False)
    out = grouped.agg(
        samples=("resolution_label", "size"),
        pct_breakout_success=("resolution_label", lambda s: (s == "breakout_success").mean()),
        pct_breakout_failure=("resolution_label", lambda s: (s == "breakout_failure").mean()),
        pct_reversal=("resolution_label", lambda s: (s == "reversal").mean()),
        pct_no_resolution=("resolution_label", lambda s: (s == "no_resolution").mean()),
        avg_target_50_gap=("target_50_gap", "mean"),
    ).reset_index()

    return out.sort_values("samples", ascending=False).reset_index(drop=True)


def build_resolution_statistics(df_seq: pd.DataFrame) -> pd.DataFrame:
    use = df_seq.dropna(subset=["resolution_label"]).copy()
    total = len(use)

    out = (
        use.groupby("resolution_label", dropna=False)
        .agg(
            samples=("resolution_label", "size"),
            avg_target_50_gap=("target_50_gap", "mean"),
            median_target_50_gap=("target_50_gap", "median"),
            winner_confirmation_rate=("winner_confirmed", "mean"),
        )
        .reset_index()
        .sort_values("samples", ascending=False)
        .reset_index(drop=True)
    )
    out["pct_of_total"] = out["samples"] / total

    # Add explicit validation metrics requested.
    absorption_present = use[use["state_absorption"] == "present"]
    absorption_reversal_rate = (
        (absorption_present["resolution_label"] == "reversal").mean()
        if len(absorption_present) > 0
        else np.nan
    )

    failure_patterns = (
        use[use["resolution_label"] == "breakout_failure"]
        .groupby(["state_key", "absorption_pattern_5"], dropna=False)
        .size()
        .reset_index(name="failure_samples")
        .sort_values("failure_samples", ascending=False)
    )

    extra = pd.DataFrame(
        {
            "resolution_label": [
                "__metric_absorption_to_reversal_rate__",
                "__metric_top_failure_pattern_count__",
            ],
            "samples": [len(absorption_present), int(failure_patterns["failure_samples"].iloc[0]) if not failure_patterns.empty else 0],
            "avg_target_50_gap": [np.nan, np.nan],
            "median_target_50_gap": [np.nan, np.nan],
            "winner_confirmation_rate": [absorption_reversal_rate, np.nan],
            "pct_of_total": [np.nan, np.nan],
        }
    )

    return pd.concat([out, extra], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build sequence-aware decision dataset from event feature states."
    )
    parser.add_argument(
        "--input",
        default="event_features_with_regime_direction_location.csv",
        help="Input CSV path (default: event_features_with_regime_direction_location.csv)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=5,
        help="Lookback window for sequence features (default: 5)",
    )
    parser.add_argument(
        "--forward",
        type=int,
        default=5,
        help="Forward window for resolution detection (default: 5)",
    )
    parser.add_argument(
        "--out-seq",
        default="event_sequence_dataset.csv",
        help="Output sequence dataset CSV path",
    )
    parser.add_argument(
        "--out-pattern",
        default="sequence_pattern_summary.csv",
        help="Output sequence pattern summary CSV path",
    )
    parser.add_argument(
        "--out-stats",
        default="resolution_statistics.csv",
        help="Output resolution statistics CSV path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    df_seq = build_sequence_dataset(df, lookback=args.lookback, forward=args.forward)
    df_pattern = build_sequence_pattern_summary(df_seq, lookback=args.lookback)
    df_stats = build_resolution_statistics(df_seq)

    Path(args.out_seq).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_pattern).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)

    df_seq.to_csv(args.out_seq, index=False)
    df_pattern.to_csv(args.out_pattern, index=False)
    df_stats.to_csv(args.out_stats, index=False)

    print("=== Sequence Dataset Build Summary ===")
    print(f"rows_input                         : {len(df)}")
    print(f"rows_output                        : {len(df_seq)}")
    print(
        "resolution_label_counts            :",
        df_seq["resolution_label"].value_counts(dropna=False).to_dict(),
    )
    print(
        "context_valid_count                :",
        int(df_seq["context_valid"].fillna(False).sum()),
    )

    q1 = df_seq[
        (df_seq["context_valid"] == True)
        & (df_seq["resolution_label"] == "breakout_success")
    ]
    q2 = df_seq[
        (df_seq["context_valid"] == True)
        & (df_seq["resolution_label"] == "breakout_failure")
    ]
    q4_base = df_seq[df_seq["state_absorption"] == "present"]
    q4_rate = (
        (q4_base["resolution_label"] == "reversal").mean() if len(q4_base) > 0 else np.nan
    )

    print("\n=== Validation Questions ===")
    print(
        f"1) Aggression->continuation count   : {len(q1)} / context={int(df_seq['context_valid'].fillna(False).sum())}"
    )
    print(f"2) Aggression failure count         : {len(q2)}")

    top_fail_patterns = (
        q2.groupby([f"pressure_trend_{args.lookback}", f"absorption_pattern_{args.lookback}"])
        .size()
        .sort_values(ascending=False)
        .head(5)
    )
    print("3) Top pre-failure sequence patterns:")
    if top_fail_patterns.empty:
        print("   none")
    else:
        for idx, cnt in top_fail_patterns.items():
            print(f"   {idx}: {int(cnt)}")

    print(f"4) Absorption->reversal rate        : {q4_rate:.4f}")

    print("\nSaved outputs:")
    print(f"- {args.out_seq}")
    print(f"- {args.out_pattern}")
    print(f"- {args.out_stats}")


if __name__ == "__main__":
    main()
