import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SignalState:
    active: bool = False
    direction: str | None = None  # long | short
    signal_price: float | None = None
    signal_extreme: float | None = None  # dynamic high for long, low for short
    frozen_extreme: float | None = None
    in_pullback: bool = False
    pullback_low: float | None = None
    pullback_high: float | None = None


@dataclass
class PositionState:
    in_position: bool = False
    direction: str | None = None
    entry_idx: int | None = None
    entry_time: int | None = None
    entry_price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    state_regime: str | None = None
    state_pressure: str | None = None
    pullback_depth_at_entry: float | None = None
    delta_norm_at_entry: float | None = None
    mae: float = 0.0
    mfe: float = 0.0


def build_mvp_trades(
    df: pd.DataFrame,
    atr_window: int = 50,
    delta_std_window: int = 50,
    pressure_quantile: float = 0.60,
    pullback_atr_min: float = 0.5,
    pullback_atr_max: float = 1.5,
    delta_norm_threshold: float = 1.5,
    price_norm_threshold: float = 0.25,
    max_hold_events: int = 80,
) -> pd.DataFrame:
    use = df.copy().reset_index(drop=True)

    # ATR proxy from event-level prices.
    use["tr"] = use["price"].diff().abs()
    use["atr"] = use["tr"].rolling(atr_window, min_periods=atr_window).mean()

    # Signed and normalized event delta for unit-consistent re-acceleration checks.
    use["delta_std"] = use["delta"].rolling(delta_std_window, min_periods=delta_std_window).std()
    use["delta_norm"] = use["delta"] / use["delta_std"].replace(0, np.nan)

    # Pressure threshold from trend regime only (MVP simplification).
    trend_abs_pressure = use.loc[use["regime"] == "trend", "pressure_50"].abs().dropna()
    if trend_abs_pressure.empty:
        raise ValueError("No trend rows found to derive pressure threshold.")
    pressure_threshold = float(trend_abs_pressure.quantile(pressure_quantile))

    signal = SignalState()
    pos = PositionState()
    trades: list[dict] = []

    for i, row in use.iterrows():
        price = float(row["price"])
        atr = row["atr"]
        delta_norm = row["delta_norm"]
        regime = str(row["regime"])
        pressure = float(row["pressure_50"])
        ts = int(row["timestamp"])

        if pd.isna(atr) or pd.isna(delta_norm):
            continue

        prev_price = float(use.loc[i - 1, "price"]) if i > 0 else price
        price_norm = (price - prev_price) / float(atr)

        # ----- manage active position -----
        if pos.in_position:
            assert pos.entry_price is not None
            assert pos.direction is not None

            if pos.direction == "long":
                pos.mae = min(pos.mae, price - pos.entry_price)
                pos.mfe = max(pos.mfe, price - pos.entry_price)
                hit_stop = price <= float(pos.stop_price)
                hit_target = price >= float(pos.target_price)
            else:
                pos.mae = min(pos.mae, pos.entry_price - price)
                pos.mfe = max(pos.mfe, pos.entry_price - price)
                hit_stop = price >= float(pos.stop_price)
                hit_target = price <= float(pos.target_price)

            held = i - int(pos.entry_idx)
            time_exit = held >= max_hold_events

            if hit_stop or hit_target or time_exit:
                pnl = (price - pos.entry_price) if pos.direction == "long" else (pos.entry_price - price)
                trades.append(
                    {
                        "entry_idx": pos.entry_idx,
                        "exit_idx": i,
                        "entry_timestamp": pos.entry_time,
                        "exit_timestamp": ts,
                        "direction": pos.direction,
                        "state_regime": pos.state_regime,
                        "state_pressure": pos.state_pressure,
                        "entry_price": pos.entry_price,
                        "exit_price": price,
                        "stop_price": pos.stop_price,
                        "target_price": pos.target_price,
                        "pnl": pnl,
                        "pnl_r": pnl / float(atr) if atr > 0 else np.nan,
                        "pullback_depth_at_entry": pos.pullback_depth_at_entry,
                        "delta_norm_at_entry": pos.delta_norm_at_entry,
                        "mae": pos.mae,
                        "mfe": pos.mfe,
                        "duration_events": held,
                        "exit_reason": "stop" if hit_stop else ("target" if hit_target else "timeout"),
                    }
                )
                pos = PositionState()
                signal = SignalState()
                continue

        # ----- trigger signal (trend-only, pressure-only) -----
        if not signal.active and not pos.in_position:
            if regime == "trend" and abs(pressure) >= pressure_threshold:
                direction = "long" if pressure > 0 else "short"
                signal = SignalState(
                    active=True,
                    direction=direction,
                    signal_price=price,
                    signal_extreme=price,
                    in_pullback=False,
                    pullback_low=price,
                    pullback_high=price,
                )
            continue

        if signal.active and not pos.in_position:
            assert signal.direction is not None
            assert signal.signal_extreme is not None

            # Update dynamic extreme until pullback begins; then freeze.
            if not signal.in_pullback:
                if signal.direction == "long":
                    signal.signal_extreme = max(float(signal.signal_extreme), price)
                    if price < float(signal.signal_extreme):
                        signal.in_pullback = True
                        signal.frozen_extreme = float(signal.signal_extreme)
                        signal.pullback_low = price
                        signal.pullback_high = float(signal.frozen_extreme)
                else:
                    signal.signal_extreme = min(float(signal.signal_extreme), price)
                    if price > float(signal.signal_extreme):
                        signal.in_pullback = True
                        signal.frozen_extreme = float(signal.signal_extreme)
                        signal.pullback_high = price
                        signal.pullback_low = float(signal.frozen_extreme)
                continue

            assert signal.frozen_extreme is not None

            # Track pullback structure after pullback phase starts.
            if signal.direction == "long":
                signal.pullback_low = min(float(signal.pullback_low), price)
                depth_atr = (float(signal.frozen_extreme) - price) / float(atr)
                pullback_mid = (float(signal.frozen_extreme) + float(signal.pullback_low)) / 2.0

                depth_ok = pullback_atr_min <= depth_atr <= pullback_atr_max
                reaccel_ok = (delta_norm > delta_norm_threshold) and (price_norm > price_norm_threshold)
                direction_gate = price > pullback_mid
                if depth_ok and reaccel_ok and direction_gate:
                    entry = price
                    structure_stop = float(signal.pullback_low)
                    vol_stop = entry - float(atr)
                    stop = min(vol_stop, structure_stop)
                    target = entry + 2.0 * float(atr)
                    pos = PositionState(
                        in_position=True,
                        direction="long",
                        entry_idx=i,
                        entry_time=ts,
                        entry_price=entry,
                        stop_price=stop,
                        target_price=target,
                        state_regime=regime,
                        state_pressure="strong_up",
                        pullback_depth_at_entry=depth_atr,
                        delta_norm_at_entry=float(delta_norm),
                        mae=0.0,
                        mfe=0.0,
                    )
            else:
                signal.pullback_high = max(float(signal.pullback_high), price)
                depth_atr = (price - float(signal.frozen_extreme)) / float(atr)
                pullback_mid = (float(signal.frozen_extreme) + float(signal.pullback_high)) / 2.0

                depth_ok = pullback_atr_min <= depth_atr <= pullback_atr_max
                reaccel_ok = (delta_norm < -delta_norm_threshold) and (price_norm < -price_norm_threshold)
                direction_gate = price < pullback_mid
                if depth_ok and reaccel_ok and direction_gate:
                    entry = price
                    structure_stop = float(signal.pullback_high)
                    vol_stop = entry + float(atr)
                    stop = max(vol_stop, structure_stop)
                    target = entry - 2.0 * float(atr)
                    pos = PositionState(
                        in_position=True,
                        direction="short",
                        entry_idx=i,
                        entry_time=ts,
                        entry_price=entry,
                        stop_price=stop,
                        target_price=target,
                        state_regime=regime,
                        state_pressure="strong_down",
                        pullback_depth_at_entry=depth_atr,
                        delta_norm_at_entry=float(delta_norm),
                        mae=0.0,
                        mfe=0.0,
                    )

            # reset stale signals if trend context disappears
            if regime != "trend":
                signal = SignalState()

    return pd.DataFrame(trades), pressure_threshold


def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            [
                {
                    "trades": 0,
                    "win_rate": np.nan,
                    "avg_pnl": np.nan,
                    "expectancy": np.nan,
                    "avg_mae": np.nan,
                    "avg_mfe": np.nan,
                    "avg_duration": np.nan,
                }
            ]
        )

    wins = trades["pnl"] > 0
    summary = {
        "trades": int(len(trades)),
        "win_rate": float(wins.mean()),
        "avg_pnl": float(trades["pnl"].mean()),
        "expectancy": float(trades["pnl"].mean()),
        "avg_mae": float(trades["mae"].mean()),
        "avg_mfe": float(trades["mfe"].mean()),
        "avg_duration": float(trades["duration_events"].mean()),
    }
    return pd.DataFrame([summary])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MVP execution backtest: trend+pressure -> pullback -> re-acceleration."
    )
    parser.add_argument(
        "--input",
        default="event_features_with_regime_direction_location.csv",
        help="Input features CSV",
    )
    parser.add_argument("--out-trades", default="mvp_trade_log.csv", help="Output trade log CSV")
    parser.add_argument("--out-summary", default="mvp_trade_summary.csv", help="Output summary CSV")
    parser.add_argument("--pressure-quantile", type=float, default=0.60)
    parser.add_argument("--pullback-min", type=float, default=0.5)
    parser.add_argument("--pullback-max", type=float, default=1.5)
    parser.add_argument("--delta-norm-threshold", type=float, default=1.5)
    parser.add_argument("--price-norm-threshold", type=float, default=0.25)
    parser.add_argument("--max-hold-events", type=int, default=80)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(input_path)
    required = {"timestamp", "price", "delta", "pressure_50", "regime"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    trades, pressure_threshold = build_mvp_trades(
        df,
        pressure_quantile=args.pressure_quantile,
        pullback_atr_min=args.pullback_min,
        pullback_atr_max=args.pullback_max,
        delta_norm_threshold=args.delta_norm_threshold,
        price_norm_threshold=args.price_norm_threshold,
        max_hold_events=args.max_hold_events,
    )

    summary = summarize_trades(trades)

    Path(args.out_trades).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(args.out_trades, index=False)
    summary.to_csv(args.out_summary, index=False)

    print("=== MVP Execution Backtest Summary ===")
    print(f"input_rows                 : {len(df)}")
    print(f"pressure_threshold_abs_q60 : {pressure_threshold:.8f}")
    print(f"trades                     : {len(trades)}")
    if len(trades) > 0:
        print(f"win_rate                   : {summary.loc[0, 'win_rate']:.4f}")
        print(f"avg_pnl                    : {summary.loc[0, 'avg_pnl']:.8f}")
        print(f"avg_mae                    : {summary.loc[0, 'avg_mae']:.8f}")
        print(f"avg_mfe                    : {summary.loc[0, 'avg_mfe']:.8f}")
        print(f"avg_duration_events        : {summary.loc[0, 'avg_duration']:.2f}")

    print("\nSaved outputs:")
    print(f"- {args.out_trades}")
    print(f"- {args.out_summary}")


if __name__ == "__main__":
    main()
