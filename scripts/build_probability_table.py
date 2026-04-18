import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "pressure_bin",
    "speed_bin",
    "absorption_flag",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build conditional probability table from event feature dataset."
    )
    parser.add_argument(
        "--input",
        default="event_features_dataset.csv",
        help="Input event feature CSV (default: event_features_dataset.csv)",
    )
    parser.add_argument(
        "--output",
        default="probability_table.csv",
        help="Output probability table CSV (default: probability_table.csv)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10_000,
        help="Minimum group sample size filter (default: 10000)",
    )
    parser.add_argument(
        "--target-col",
        default="target_direction_50",
        help="Target direction column to model (default: target_direction_50)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    required_cols = REQUIRED_COLUMNS + [args.target_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Step 1: enforce no missing values on required feature/target columns.
    before = len(df)
    df = df.dropna(subset=required_cols).copy()
    dropped = before - len(df)

    # Step 3: grouped conditional probability table.
    result = (
        df.groupby(["pressure_bin", "speed_bin", "absorption_flag"], dropna=False)[
            args.target_col
        ]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "prob_up", "count": "samples"})
    )

    # Step 4: low-sample filter.
    result = result[result["samples"] >= args.min_samples].copy()

    # Step 5 + Step 6.
    result = result.sort_values("prob_up", ascending=False).reset_index(drop=True)
    result["edge"] = (result["prob_up"] - 0.5).abs()

    # Step 7 final output columns + sort by edge descending.
    result = result[
        [
            "pressure_bin",
            "speed_bin",
            "absorption_flag",
            "prob_up",
            "edge",
            "samples",
        ]
    ].sort_values("edge", ascending=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    # Step 8 validation.
    print("=== Probability Table Build Summary ===")
    print(f"target_col                 : {args.target_col}")
    print(f"input_rows                 : {before}")
    print(f"rows_dropped_missing       : {dropped}")
    print(f"rows_used                  : {len(df)}")
    print(f"groups_after_filter        : {len(result)}")

    if len(result) > 0:
        print(f"min_prob_up                : {result['prob_up'].min():.8f}")
        print(f"max_prob_up                : {result['prob_up'].max():.8f}")
        print("samples_distribution_desc  :")
        print(result["samples"].describe().to_string())
    else:
        print("min_prob_up                : n/a")
        print("max_prob_up                : n/a")
        print("samples_distribution_desc  : n/a")

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
