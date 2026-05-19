import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None

from mvp_execution_backtest import (
    build_exit_reason_stats,
    build_mvp_trades,
    summarize_trades,
)


CSV_COLUMNS = [
    "trade_id",
    "price",
    "quantity",
    "quote_qty",
    "timestamp",
    "is_buyer_maker",
]

PARQUET_REQUIRED_COLUMNS = ["price", "qty", "timestamp", "is_buyer_maker"]


def _causal_tertile_bins(
    series: pd.Series, labels: list[str], min_history: int = 100
) -> pd.Series:
    hist = series.shift(1)
    q1 = hist.expanding(min_periods=min_history).quantile(1 / 3)
    q2 = hist.expanding(min_periods=min_history).quantile(2 / 3)

    out = pd.Series(index=series.index, dtype="object")
    out.loc[series <= q1] = labels[0]
    out.loc[(series > q1) & (series <= q2)] = labels[1]
    out.loc[series > q2] = labels[2]
    return out


def stream_core_features(
    files: list[Path],
    chunk_size: int = 2_000_000,
    subsample_step: int = 50,
    progress_every_chunks: int = 1,
) -> tuple[pd.DataFrame, dict]:
    carry = pd.DataFrame(columns=["timestamp", "price", "quantity", "delta"])

    out_timestamp: list[int] = []
    out_price: list[float] = []
    out_delta: list[float] = []
    out_pressure50: list[float] = []
    out_speed50: list[float] = []
    out_range50: list[float] = []
    out_volume50: list[float] = []

    total_rows = 0
    sampled_rows = 0
    ts_disorder_count = 0
    non_positive_price = 0
    non_positive_qty = 0
    bad_timestamp = 0
    boundary_checks = []

    prev_ts = None

    def _iter_trade_chunks(fp: Path):
        """Yield normalized trade chunks with columns: price, quantity, timestamp, is_buyer_maker."""
        if fp.suffix.lower() == ".parquet":
            if pq is None:
                raise ImportError("pyarrow is required to read parquet input files.")
            pf = pq.ParquetFile(fp)
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg, columns=PARQUET_REQUIRED_COLUMNS)
                chunk = table.to_pandas()
                chunk = chunk.rename(columns={"qty": "quantity"})
                yield chunk
        else:
            for chunk in pd.read_csv(
                fp,
                header=None,
                names=CSV_COLUMNS,
                usecols=["price", "quantity", "timestamp", "is_buyer_maker"],
                chunksize=chunk_size,
            ):
                yield chunk

    def _fmt_ts(ms: int | None) -> str:
        if ms is None:
            return "n/a"
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()

    for fp in files:
        print(f"\n[PROGRESS] Starting file: {fp}")
        file_first_ts = None
        file_last_ts = None
        chunk_idx = 0
        for chunk in _iter_trade_chunks(fp):
            chunk_idx += 1
            price = pd.to_numeric(chunk["price"], errors="coerce")
            qty = pd.to_numeric(chunk["quantity"], errors="coerce")
            ts = pd.to_numeric(chunk["timestamp"], errors="coerce")

            maker_raw = chunk["is_buyer_maker"]
            maker_flag = maker_raw.astype(str).str.lower().map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                }
            )

            valid = (
                price.notna()
                & qty.notna()
                & ts.notna()
                & np.isfinite(price)
                & np.isfinite(qty)
                & np.isfinite(ts)
                & (price > 0)
                & (qty > 0)
                & maker_flag.notna()
            )

            non_positive_price += int(((price <= 0) & price.notna()).sum())
            non_positive_qty += int(((qty <= 0) & qty.notna()).sum())
            bad_timestamp += int((~ts.notna()).sum())

            if not valid.any():
                total_rows += len(chunk)
                if progress_every_chunks > 0 and chunk_idx % progress_every_chunks == 0:
                    print(
                        f"[PROGRESS] {fp.name} chunk={chunk_idx} valid_rows=0 "
                        f"global_rows={total_rows} last_ts={_fmt_ts(prev_ts)}"
                    )
                continue

            p = price.loc[valid].astype("float64")
            q = qty.loc[valid].astype("float64")
            t = ts.loc[valid].astype("int64")
            m = maker_flag.loc[valid].astype(bool)
            d = pd.Series(np.where(m, -q, q), index=t.index, dtype="float64")

            valid_df = pd.DataFrame(
                {
                    "timestamp": t.values,
                    "price": p.values,
                    "quantity": q.values,
                    "delta": d.values,
                }
            )

            if not valid_df.empty:
                if prev_ts is not None and int(valid_df.iloc[0]["timestamp"]) < prev_ts:
                    ts_disorder_count += 1
                ts_disorder_count += int((valid_df["timestamp"].diff() < 0).sum())
                prev_ts = int(valid_df.iloc[-1]["timestamp"])

                if file_first_ts is None:
                    file_first_ts = int(valid_df.iloc[0]["timestamp"])
                file_last_ts = int(valid_df.iloc[-1]["timestamp"])

            start_row_idx = total_rows
            n_valid = len(valid_df)
            total_rows += n_valid

            if n_valid == 0:
                continue

            combined = pd.concat([carry, valid_df], ignore_index=True)

            delta_50 = combined["delta"].rolling(window=50, min_periods=50).sum()
            volume_50 = combined["quantity"].rolling(window=50, min_periods=50).sum()
            pressure_50 = delta_50 / volume_50.replace(0, np.nan)

            rolling_max = combined["price"].rolling(window=50, min_periods=50).max()
            rolling_min = combined["price"].rolling(window=50, min_periods=50).min()
            range_50 = rolling_max - rolling_min

            dt_seconds = ((combined["timestamp"] - combined["timestamp"].shift(50)) / 1000.0).to_numpy(
                dtype=np.float64
            )
            safe_dt = np.where(dt_seconds >= 0.001, dt_seconds, np.nan)
            speed_50 = 50.0 / safe_dt
            speed_50 = np.clip(speed_50, None, 1000.0)

            part = combined.iloc[len(carry) :].copy()
            part["pressure_50"] = pressure_50.iloc[len(carry) :].to_numpy()
            part["volume_50"] = volume_50.iloc[len(carry) :].to_numpy()
            part["range_50"] = range_50.iloc[len(carry) :].to_numpy()
            part["speed_50"] = speed_50[len(carry) :]

            global_positions_1based = np.arange(start_row_idx + 1, start_row_idx + n_valid + 1)
            subsample_mask = (global_positions_1based % subsample_step) == 0
            warm_mask = (
                part["pressure_50"].notna()
                & part["volume_50"].notna()
                & part["range_50"].notna()
                & pd.Series(part["speed_50"]).notna()
            )
            take = subsample_mask & warm_mask.to_numpy()

            if take.any():
                picked = part.iloc[np.where(take)[0]]
                sampled_rows += len(picked)

                out_timestamp.extend(picked["timestamp"].astype("int64").tolist())
                out_price.extend(picked["price"].astype("float64").tolist())
                out_delta.extend(picked["delta"].astype("float64").tolist())
                out_pressure50.extend(picked["pressure_50"].astype("float64").tolist())
                out_speed50.extend(pd.to_numeric(picked["speed_50"], errors="coerce").astype("float64").tolist())
                out_range50.extend(picked["range_50"].astype("float64").tolist())
                out_volume50.extend(picked["volume_50"].astype("float64").tolist())

            carry = combined.tail(50).copy()

            if progress_every_chunks > 0 and chunk_idx % progress_every_chunks == 0:
                chunk_last_ts = int(valid_df.iloc[-1]["timestamp"]) if not valid_df.empty else prev_ts
                print(
                    f"[PROGRESS] {fp.name} chunk={chunk_idx} valid_rows={n_valid} "
                    f"sampled_rows={sampled_rows} global_rows={total_rows} "
                    f"last_ts_ms={chunk_last_ts} last_ts_utc={_fmt_ts(chunk_last_ts)}"
                )

        boundary_checks.append((fp.name, file_first_ts, file_last_ts))
        print(
            f"[PROGRESS] Completed file: {fp.name} "
            f"first_ts_ms={file_first_ts} last_ts_ms={file_last_ts} "
            f"first_ts_utc={_fmt_ts(file_first_ts)} last_ts_utc={_fmt_ts(file_last_ts)}"
        )

    df = pd.DataFrame(
        {
            "timestamp": out_timestamp,
            "price": out_price,
            "delta": out_delta,
            "pressure_50": out_pressure50,
            "speed_50": out_speed50,
            "range_50": out_range50,
            "volume_50": out_volume50,
        }
    )

    integrity = {
        "total_valid_rows_processed": total_rows,
        "sampled_rows": sampled_rows,
        "timestamp_disorder_count": ts_disorder_count,
        "non_positive_price_rows": non_positive_price,
        "non_positive_qty_rows": non_positive_qty,
        "bad_timestamp_rows": bad_timestamp,
        "file_boundaries": boundary_checks,
    }
    return df, integrity


def build_regime_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["vol_p80_50"] = out["volume_50"].shift(1).expanding(min_periods=50).quantile(0.80)
    out["range_p30_50"] = out["range_50"].shift(1).expanding(min_periods=50).quantile(0.30)
    out["absorption_flag"] = (
        (out["volume_50"] > out["vol_p80_50"]) & (out["range_50"] < out["range_p30_50"])
    ).astype("int64")

    out["pressure_bin"] = _causal_tertile_bins(
        out["pressure_50"], ["low", "mid", "high"], min_history=100
    )
    out["speed_bin"] = _causal_tertile_bins(
        out["speed_50"], ["slow", "medium", "fast"], min_history=100
    )

    trend_mask = (
        out["absorption_flag"].eq(0)
        & out["pressure_bin"].isin(["high", "low"])
        & out["speed_bin"].isin(["fast", "medium"])
    )
    out["regime"] = np.select(
        [out["absorption_flag"].eq(1), trend_mask],
        ["conflict", "trend"],
        default="neutral",
    )

    out = out.dropna(subset=["pressure_50", "speed_50", "range_50", "volume_50", "pressure_bin", "speed_bin"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run execution backtest for BTCUSDT Apr-Jun 2021 as continuous data.")
    parser.add_argument(
        "--file-apr",
        default="storage/btcusdt/BTCUSDT-trades-2021-04.parquet",
    )
    parser.add_argument(
        "--file-may",
        default="storage/btcusdt/BTCUSDT-trades-2021-05.parquet",
    )
    parser.add_argument(
        "--file-jun",
        default="storage/btcusdt/BTCUSDT-trades-2021-06.parquet",
    )
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--subsample-step", type=int, default=50)
    parser.add_argument(
        "--progress-every-chunks",
        type=int,
        default=1,
        help="Print progress every N chunks (default: 1)",
    )
    parser.add_argument("--out-input", default="event_features_with_regime_direction_location_202104_202106.parquet")
    parser.add_argument("--out-trades", default="mvp_trade_log_202104_202106.csv")
    parser.add_argument("--out-summary", default="mvp_trade_summary_202104_202106.csv")
    parser.add_argument("--out-exit-stats", default="mvp_exit_reason_stats_202104_202106.csv")
    parser.add_argument("--fee-rate", type=float, default=0.0004)
    args = parser.parse_args()

    files = [Path(args.file_apr), Path(args.file_may), Path(args.file_jun)]
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"Missing input file: {f}")

    core_df, integrity = stream_core_features(
        files,
        chunk_size=args.chunk_size,
        subsample_step=args.subsample_step,
        progress_every_chunks=args.progress_every_chunks,
    )
    print("\n[STAGE] Finished streaming raw data → core_df")
    print(f"[STAGE] core_df rows: {len(core_df)}")

    print("[STAGE] Building regime columns...")
    regime_df = build_regime_columns(core_df)
    print(f"[STAGE] regime_df rows: {len(regime_df)}")

    print("[STAGE] Starting backtest (build_mvp_trades)...")
    backtest_input = regime_df[["timestamp", "price", "delta", "pressure_50", "regime"]].copy()
    trades, pressure_threshold = build_mvp_trades(backtest_input, fee_rate=args.fee_rate)
    if not trades.empty:
        trades["cum_pnl"] = trades["pnl"].cumsum()

    summary = summarize_trades(trades)
    exit_stats = build_exit_reason_stats(trades)

    Path(args.out_input).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_trades).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_exit_stats).parent.mkdir(parents=True, exist_ok=True)

    regime_df.to_parquet(args.out_input, index=False)
    trades.to_csv(args.out_trades, index=False)
    summary.to_csv(args.out_summary, index=False)
    exit_stats.to_csv(args.out_exit_stats, index=False)

    print("=== Continuous Apr-Jun 2021 Backtest Run ===")
    print(f"input_files                     : {[str(f) for f in files]}")
    print(f"rows_processed_valid            : {integrity['total_valid_rows_processed']}")
    print(f"rows_sampled                    : {integrity['sampled_rows']}")
    print(f"rows_after_regime_filtering     : {len(regime_df)}")
    print(f"timestamp_disorder_count        : {integrity['timestamp_disorder_count']}")
    print(f"non_positive_price_rows         : {integrity['non_positive_price_rows']}")
    print(f"non_positive_qty_rows           : {integrity['non_positive_qty_rows']}")
    print(f"bad_timestamp_rows              : {integrity['bad_timestamp_rows']}")
    print(f"pressure_threshold_abs_q50      : {pressure_threshold:.8f}")
    print(f"trades                          : {len(trades)}")
    if not trades.empty:
        print(f"cumulative_pnl                  : {trades['cum_pnl'].iloc[-1]:.8f}")

    print("\nFile boundaries (first_ts, last_ts):")
    for name, first_ts, last_ts in integrity["file_boundaries"]:
        print(f"- {name}: {first_ts} -> {last_ts}")

    print("\nSaved outputs:")
    print(f"- {args.out_input}")
    print(f"- {args.out_trades}")
    print(f"- {args.out_summary}")
    print(f"- {args.out_exit_stats}")


if __name__ == "__main__":
    main()
