import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_REQUIRED = ["pressure_bin", "speed_bin", "absorption_flag", "location_bin"]
TARGET_CANDIDATES = ["target_direction_50_gap", "target_direction_50"]


def choose_target_column(df: pd.DataFrame, preferred: str | None) -> str:
    if preferred:
        if preferred not in df.columns:
            raise ValueError(f"Preferred target column not found: {preferred}")
        return preferred

    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c

    raise ValueError(
        "No valid target direction column found. Expected one of: "
        + ", ".join(TARGET_CANDIDATES)
    )


def build_direction(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [df["pressure_bin"].eq("high"), df["pressure_bin"].eq("low")],
            ["up", "down"],
            default="none",
        ),
        index=df.index,
        name="direction",
    )


def build_regime(df: pd.DataFrame) -> pd.Series:
    trend_mask = (
        df["absorption_flag"].eq(0)
        & df["pressure_bin"].isin(["high", "low"])
        & df["speed_bin"].isin(["fast", "medium"])
    )

    return pd.Series(
        np.select(
            [df["absorption_flag"].eq(1), trend_mask],
            ["conflict", "trend"],
            default="neutral",
        ),
        index=df.index,
        name="regime",
    )


def probability_table(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    min_samples: int | None = None,
) -> pd.DataFrame:
    out = (
        df.groupby(group_cols, dropna=False)[target_col]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "prob_up", "count": "samples"})
    )
    out["edge"] = (out["prob_up"] - 0.5).abs()
    if min_samples is not None:
        out = out[out["samples"] >= min_samples].copy()
    return out.sort_values("edge", ascending=False).reset_index(drop=True)


def duration_stats(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df[["regime"]].copy()
    tmp["regime_change"] = tmp["regime"].ne(tmp["regime"].shift(1)).astype("int64")
    tmp["regime_id"] = tmp["regime_change"].cumsum()
    runs = tmp.groupby(["regime_id", "regime"]).size().reset_index(name="duration")
    return runs.groupby("regime")["duration"].describe().reset_index()


def transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    next_regime = df["regime"].shift(-1)
    mat = pd.crosstab(df["regime"], next_regime, normalize="index")
    return mat


def best_condition(
    table: pd.DataFrame,
    regime: str,
    direction: str,
    location_bin: str,
    absorption_flag: int,
):
    mask = (
        table["regime"].eq(regime)
        & table["direction"].eq(direction)
        & table["location_bin"].eq(location_bin)
        & table["absorption_flag"].eq(absorption_flag)
    )
    subset = table[mask]
    if subset.empty:
        return None
    return subset.sort_values("edge", ascending=False).iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build regime-direction-location outputs from event features dataset."
    )
    parser.add_argument(
        "--input",
        default="event_features_dataset.csv",
        help="Input event features CSV (default: event_features_dataset.csv)",
    )
    parser.add_argument(
        "--target-col",
        default=None,
        help="Optional target direction column override (e.g. target_direction_50 or target_direction_50_gap)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    target_col = choose_target_column(df, args.target_col)
    required = BASE_REQUIRED + [target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    before = len(df)
    df = df.dropna(subset=required).copy()

    # Extra safety for flat labels if source data contains them.
    if "target_50_gap" in df.columns and target_col == "target_direction_50_gap":
        df = df[df["target_50_gap"] != 0].copy()
    elif "target_50" in df.columns and target_col == "target_direction_50":
        df = df[df["target_50"] != 0].copy()

    dropped = before - len(df)

    # Vectorized direction + regime.
    df["direction"] = build_direction(df)
    df["regime"] = build_regime(df)

    # Primary validation table: regime + direction.
    regime_direction_prob = probability_table(
        df, ["regime", "direction"], target_col, min_samples=None
    )

    # Breakout vs failed breakout context table.
    breakout_failure = probability_table(
        df,
        ["regime", "direction", "location_bin", "absorption_flag"],
        target_col,
        min_samples=10_000,
    )

    # Regime dynamics.
    regime_duration = duration_stats(df)
    regime_transition = transition_matrix(df)

    # Persist full tagged dataset.
    df.to_csv("event_features_with_regime_direction_location.csv", index=False)
    regime_direction_prob.to_csv("regime_direction_probability.csv", index=False)
    breakout_failure.to_csv("breakout_failure_table.csv", index=False)
    regime_duration.to_csv("regime_duration.csv", index=False)
    regime_transition.to_csv("regime_transition.csv")

    # Validation summary outputs.
    edge_restored_up = regime_direction_prob[
        (regime_direction_prob["regime"] == "trend")
        & (regime_direction_prob["direction"] == "up")
    ]
    edge_restored_down = regime_direction_prob[
        (regime_direction_prob["regime"] == "trend")
        & (regime_direction_prob["direction"] == "down")
    ]

    breakout_cond = best_condition(
        breakout_failure,
        regime="trend",
        direction="up",
        location_bin="high",
        absorption_flag=0,
    )
    failed_breakout_cond = best_condition(
        breakout_failure,
        regime="trend",
        direction="up",
        location_bin="high",
        absorption_flag=1,
    )

    regime_dist = (
        df["regime"].value_counts().rename_axis("regime").reset_index(name="samples")
    )
    regime_dist["pct"] = 100.0 * regime_dist["samples"] / len(df)

    regime_dir_dist = (
        df.groupby(["regime", "direction"]).size().reset_index(name="samples")
    )
    regime_dir_dist["pct"] = 100.0 * regime_dir_dist["samples"] / len(df)

    self_persistence = {}
    for reg in regime_transition.index:
        self_persistence[reg] = float(regime_transition.loc[reg, reg]) if reg in regime_transition.columns else np.nan

    print("=== Regime Direction Pipeline Summary ===")
    print(f"target_col_used                  : {target_col}")
    print(f"rows_input                       : {before}")
    print(f"rows_dropped_missing_or_flat     : {dropped}")
    print(f"rows_used                        : {len(df)}")

    print("\nEdge restoration check (trend split by direction):")
    if not edge_restored_up.empty:
        r = edge_restored_up.iloc[0]
        print(
            f"trend+up                          prob_up={r['prob_up']:.6f} samples={int(r['samples'])} edge={r['edge']:.6f}"
        )
    else:
        print("trend+up                          not found")

    if not edge_restored_down.empty:
        r = edge_restored_down.iloc[0]
        print(
            f"trend+down                        prob_up={r['prob_up']:.6f} samples={int(r['samples'])} edge={r['edge']:.6f}"
        )
    else:
        print("trend+down                        not found")

    print("\nBest breakout condition (trend/up/high/no absorption):")
    if breakout_cond is not None:
        print(
            f"prob_up={breakout_cond['prob_up']:.6f}, samples={int(breakout_cond['samples'])}, edge={breakout_cond['edge']:.6f}"
        )
    else:
        print("No qualifying breakout condition with samples >= 10,000")

    print("\nBest failed breakout condition (trend/up/high/absorption):")
    if failed_breakout_cond is not None:
        print(
            f"prob_up={failed_breakout_cond['prob_up']:.6f}, samples={int(failed_breakout_cond['samples'])}, edge={failed_breakout_cond['edge']:.6f}"
        )
    else:
        print("No qualifying failed-breakout condition with samples >= 10,000")

    print("\nRegime persistence (self-transition rates):")
    print(self_persistence)

    print("\nSample distribution by regime:")
    print(regime_dist.to_string(index=False))

    print("\nSample distribution by regime x direction:")
    print(regime_dir_dist.sort_values(["regime", "direction"]).to_string(index=False))

    print("\nSaved outputs:")
    print("- event_features_with_regime_direction_location.csv")
    print("- regime_direction_probability.csv")
    print("- breakout_failure_table.csv")
    print("- regime_duration.csv")
    print("- regime_transition.csv")


if __name__ == "__main__":
    main()
