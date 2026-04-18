import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "price",
    "pressure_50",
    "pressure_bin",
    "speed_50",
    "absorption_flag",
    "location_bin",
    "regime",
]


def pressure_state_from_bin(pressure_bin: str) -> str:
    if pressure_bin == "high":
        return "up"
    if pressure_bin == "low":
        return "down"
    return "neutral"


def opposite(direction: str | None) -> str | None:
    if direction == "up":
        return "down"
    if direction == "down":
        return "up"
    return None


@dataclass
class TradeContext:
    direction: str | None = None
    entry_price: float | None = None
    stop_price: float | None = None
    events_in_trade: int = 0


class RealTimeStateMachine:
    def __init__(
        self,
        window_size: int = 100,
        max_trade_events: int = 80,
        stop_fallback_pct: float = 0.001,
        context_direction_fallback_events: int = 3,
        zone_max_age_events: int = 50,
        confirmed_timeout_events: int = 15,
        price_move_threshold_pct: float = 0.00005,
    ) -> None:
        self.window_size = window_size
        self.max_trade_events = max_trade_events
        self.stop_fallback_pct = stop_fallback_pct
        self.context_direction_fallback_events = context_direction_fallback_events
        self.zone_max_age_events = zone_max_age_events
        self.confirmed_timeout_events = confirmed_timeout_events
        self.price_move_threshold_pct = price_move_threshold_pct

        self.state = "IDLE"
        self.in_trade = False

        self.context_direction: str | None = None
        self.resolution_type = "none"
        self.confirmed_direction: str | None = None

        self.prev_pressure_state: str | None = None
        self.prev_absorption_state: str | None = None
        self.prev_price: float | None = None
        self.prev_pressure_value: float | None = None

        # Short sequence memory (cleared on state-cycle reset).
        self.pressure_history = deque(maxlen=3)
        self.absorption_history = deque(maxlen=3)
        self.resolution_history = deque(maxlen=3)

        # Rolling histories for adaptive aggression thresholds.
        self.speed_history = deque(maxlen=window_size)
        self.abs_pressure_history = deque(maxlen=window_size)

        # Structural zones (updated continuously).
        self.last_opposing_zone = {
            "up": {"price": None, "event_idx": None},
            "down": {"price": None, "event_idx": None},
        }

        self.trade = TradeContext()
        self.pressure_flip_count = 0
        self.absorption_against_count = 0
        self.event_idx = 0
        self.context_events_without_direction = 0
        self.confirmed_events_count = 0

    def _reset_cycle_state(self) -> None:
        self.state = "IDLE"
        self.context_direction = None
        self.resolution_type = "none"
        self.confirmed_direction = None
        self.pressure_history.clear()
        self.absorption_history.clear()
        self.resolution_history.clear()
        self.prev_pressure_state = None
        self.prev_absorption_state = None
        self.prev_price = None
        self.prev_pressure_value = None
        self.pressure_flip_count = 0
        self.absorption_against_count = 0
        self.context_events_without_direction = 0
        self.confirmed_events_count = 0

    def _stable_context_direction(self, current_pressure_state: str) -> str | None:
        last_pressure = None
        if len(self.pressure_history) >= 1:
            last_pressure = self.pressure_history[-1]

        if (
            last_pressure in {"up", "down"}
            and current_pressure_state in {"up", "down"}
            and last_pressure == current_pressure_state
        ):
            return current_pressure_state
        return None

    def _context_valid(self, regime: str, location: str, pressure_state: str) -> bool:
        return (
            regime == "trend"
            and location in {"low", "high"}
            and pressure_state in {"up", "down"}
        )

    def _update_structural_zones(self, pressure_state: str, price: float, pressure_value: float) -> None:
        if len(self.abs_pressure_history) < self.window_size:
            return

        rolling_q70_pressure = float(np.nanquantile(np.array(self.abs_pressure_history, dtype="float64"), 0.7))
        is_strong_pressure = abs(pressure_value) > rolling_q70_pressure
        if not is_strong_pressure:
            return

        if pressure_state == "down":
            self.last_opposing_zone["up"] = {"price": price, "event_idx": self.event_idx}
        elif pressure_state == "up":
            self.last_opposing_zone["down"] = {"price": price, "event_idx": self.event_idx}

    def _get_fresh_zone_price(self, direction: str) -> float | None:
        zone = self.last_opposing_zone.get(direction)
        if not isinstance(zone, dict):
            return None

        zone_price = zone.get("price")
        zone_event_idx = zone.get("event_idx")
        if zone_price is None or zone_event_idx is None:
            return None

        if (self.event_idx - int(zone_event_idx)) > self.zone_max_age_events:
            return None
        return float(zone_price)

    def _update_resolution_history(self, candidate: str) -> None:
        if not self.resolution_history:
            self.resolution_history.append(candidate)
            return

        previous_resolution = self.resolution_history[-1]
        if candidate == previous_resolution:
            self.resolution_history.append(candidate)
        else:
            self.resolution_history.clear()
            self.resolution_history.append(candidate)

    def _detect_resolution(
        self,
        current_direction: str,
        current_absorption: str,
        current_price: float,
        current_pressure_value: float,
    ) -> tuple[str, str | None]:
        if self.context_direction not in {"up", "down"}:
            return "none", None

        recent_pressure = list(self.pressure_history)
        valid_events = [p for p in recent_pressure if p != "neutral"]
        extended_events = list(valid_events)
        if current_direction != "neutral":
            extended_events.append(current_direction)
        same_side_count = sum(1 for p in extended_events if p == current_direction)

        continuation_confirmed = (
            len(extended_events) >= 2
            and current_direction in {"up", "down"}
            and same_side_count >= 2
        )

        prev_direction = self.prev_pressure_state
        prev_price = self.prev_price
        prev_pressure_value = self.prev_pressure_value

        pressure_weakening = False
        no_price_follow_through = False
        if prev_pressure_value is not None:
            pressure_weakening = abs(current_pressure_value) < abs(prev_pressure_value)
        if prev_price is not None:
            if self.context_direction == "up":
                no_price_follow_through = current_price <= prev_price
            else:
                no_price_follow_through = current_price >= prev_price

        absorption_change_against = (
            self.prev_absorption_state is not None
            and self.prev_absorption_state != current_absorption
            and current_absorption == "present"
            and current_direction == opposite(self.context_direction)
        )
        pressure_does_not_recover = (
            current_direction != self.context_direction
            and prev_direction != self.context_direction
        )

        failure_confirmed = (
            (pressure_weakening and no_price_follow_through)
            or (absorption_change_against and pressure_does_not_recover)
        )

        opposite_side = opposite(self.context_direction)
        opposite_count = sum(1 for p in valid_events if p == opposite_side)
        reversal_confirmed = (
            opposite_side is not None
            and current_direction == opposite_side
            and opposite_count >= 2
            and len(self.abs_pressure_history) >= self.window_size
            and abs(current_pressure_value)
            > float(np.nanquantile(np.array(self.abs_pressure_history, dtype="float64"), 0.7))
        )

        if reversal_confirmed:
            return "reversal", opposite_side
        if failure_confirmed:
            return "failure", opposite_side
        if continuation_confirmed:
            return "continuation", current_direction
        return "none", None

    def _aggression_spike(
        self,
        state: str,
        speed: float,
        pressure_value: float,
        pressure_direction: str,
        current_price: float,
    ) -> bool:
        if state != "CONFIRMED":
            return False

        # Warm-up guard for rolling metrics.
        if len(self.speed_history) < self.window_size:
            return False
        if len(self.abs_pressure_history) < self.window_size:
            return False

        speed_arr = np.array(self.speed_history, dtype="float64")
        abs_pressure_arr = np.array(self.abs_pressure_history, dtype="float64")

        rolling_q90_speed = float(np.nanquantile(speed_arr, 0.9))
        rolling_q80_pressure = float(np.nanquantile(abs_pressure_arr, 0.8))
        speed_floor = float(np.nanmedian(speed_arr))

        return (
            speed > max(rolling_q90_speed, speed_floor)
            and abs(pressure_value) > rolling_q80_pressure
            and pressure_direction == self.confirmed_direction
            and self.prev_price is not None
            and (
                (
                    self.confirmed_direction == "up"
                    and current_price > self.prev_price + (self.prev_price * self.price_move_threshold_pct)
                )
                or (
                    self.confirmed_direction == "down"
                    and current_price < self.prev_price - (self.prev_price * self.price_move_threshold_pct)
                )
            )
        )

    def _should_exit(self, regime: str, pressure_state: str, absorption_state: str, price: float) -> bool:
        if not self.in_trade or self.trade.direction not in {"up", "down"}:
            return False

        position = self.trade.direction
        opposite_side = opposite(position)

        pressure_flip_against = pressure_state == opposite_side
        if pressure_flip_against:
            self.pressure_flip_count += 1
        else:
            self.pressure_flip_count = 0
        pressure_flip_persistent = self.pressure_flip_count >= 2

        absorption_against = absorption_state == "present" and pressure_state == opposite_side
        if absorption_against:
            self.absorption_against_count += 1
        else:
            self.absorption_against_count = 0
        absorption_against_persistent = self.absorption_against_count >= 2
        regime_changed = regime != "trend"

        structural_stop_hit = False
        if self.trade.stop_price is not None:
            if position == "up":
                structural_stop_hit = price <= self.trade.stop_price
            else:
                structural_stop_hit = price >= self.trade.stop_price

        time_expired = self.trade.events_in_trade >= self.max_trade_events

        return (
            pressure_flip_persistent
            or absorption_against_persistent
            or regime_changed
            or structural_stop_hit
            or time_expired
        )

    def process_event(self, row: pd.Series) -> dict:
        self.event_idx += 1

        price = float(row["price"])
        speed = float(row["speed_50"])
        pressure_value = float(row["pressure_50"])
        pressure_state = pressure_state_from_bin(str(row["pressure_bin"]))
        absorption_state = "present" if int(row["absorption_flag"]) == 1 else "absent"
        location = str(row["location_bin"])
        regime = str(row["regime"])

        context_valid = self._context_valid(regime, location, pressure_state)

        # Keep structural zones updated continuously.
        self._update_structural_zones(pressure_state, price, pressure_value)

        exit_signal = 0
        entry_signal = 0
        entry_direction = "none"
        aggression_spike = False
        exited_this_event = False

        if self.in_trade:
            self.trade.events_in_trade += 1
            if self._should_exit(regime, pressure_state, absorption_state, price):
                exit_signal = 1
                self.in_trade = False
                self.trade = TradeContext()
                self._reset_cycle_state()
                exited_this_event = True

        if not self.in_trade and not exited_this_event:
            if self.state == "IDLE":
                if context_valid:
                    self.state = "CONTEXT"
                    self.context_events_without_direction = 0
                    stable_direction = self._stable_context_direction(pressure_state)
                    if stable_direction is not None:
                        self.context_direction = stable_direction
            elif self.state == "CONTEXT":
                if not context_valid:
                    self._reset_cycle_state()
                else:
                    stable_direction = self._stable_context_direction(pressure_state)
                    if stable_direction is not None:
                        self.context_direction = stable_direction
                    elif self.context_direction is None:
                        self.context_events_without_direction += 1
                        if (
                            self.context_events_without_direction
                            >= self.context_direction_fallback_events
                            and pressure_state in {"up", "down"}
                        ):
                            self.context_direction = pressure_state
                if self.context_direction in {"up", "down"} and pressure_state in {"up", "down"}:
                    self.state = "INTERACTION"
            elif self.state == "INTERACTION":
                if not context_valid:
                    self._reset_cycle_state()
                else:
                    candidate, candidate_direction = self._detect_resolution(
                        pressure_state,
                        absorption_state,
                        price,
                        pressure_value,
                    )
                    if candidate != "none":
                        self.resolution_type = candidate
                        self.confirmed_direction = candidate_direction
                        self._update_resolution_history(candidate)
                        self.state = "RESOLUTION"
            elif self.state == "RESOLUTION":
                if not context_valid:
                    self._reset_cycle_state()
                else:
                    candidate, candidate_direction = self._detect_resolution(
                        pressure_state,
                        absorption_state,
                        price,
                        pressure_value,
                    )
                    if candidate == "none":
                        self.state = "INTERACTION"
                        self.resolution_history.clear()
                        self.resolution_type = "none"
                        self.confirmed_direction = None
                    else:
                        self.resolution_type = candidate
                        self.confirmed_direction = candidate_direction
                        self._update_resolution_history(candidate)
                        if len(self.resolution_history) >= 2:
                            last_two = list(self.resolution_history)[-2:]
                            if last_two[0] == last_two[1]:
                                self.state = "CONFIRMED"
                                self.confirmed_events_count = 0
            elif self.state == "CONFIRMED":
                if not context_valid:
                    self._reset_cycle_state()
                else:
                    self.confirmed_events_count += 1
                    if self.confirmed_events_count > self.confirmed_timeout_events:
                        self.state = "INTERACTION"
                        self.resolution_history.clear()
                        self.resolution_type = "none"
                        self.confirmed_direction = None
                        self.confirmed_events_count = 0
                    else:
                        candidate, candidate_direction = self._detect_resolution(
                            pressure_state,
                            absorption_state,
                            price,
                            pressure_value,
                        )
                        if candidate == "none":
                            self.state = "INTERACTION"
                            self.resolution_history.clear()
                            self.resolution_type = "none"
                            self.confirmed_direction = None
                        elif candidate != self.resolution_type:
                            self.state = "RESOLUTION"
                            self.resolution_type = candidate
                            self.confirmed_direction = candidate_direction
                            self.resolution_history.clear()
                            self.resolution_history.append(candidate)
                            self.confirmed_events_count = 0
                        else:
                            aggression_spike = self._aggression_spike(
                                self.state,
                                speed,
                                pressure_value,
                                pressure_state,
                                price,
                            )
                            entry_signal = int(
                                self.state == "CONFIRMED"
                                and aggression_spike
                                and not self.in_trade
                            )
                            if entry_signal == 1 and self.confirmed_direction in {"up", "down"}:
                                entry_direction = self.confirmed_direction
                                self.in_trade = True
                                self.state = "IN_TRADE"
                                self.pressure_flip_count = 0
                                self.absorption_against_count = 0
                                stop_price = self._get_fresh_zone_price(entry_direction)
                                if stop_price is None:
                                    if entry_direction == "up":
                                        stop_price = price * (1.0 - self.stop_fallback_pct)
                                    else:
                                        stop_price = price * (1.0 + self.stop_fallback_pct)
                                self.trade = TradeContext(
                                    direction=entry_direction,
                                    entry_price=price,
                                    stop_price=stop_price,
                                    events_in_trade=0,
                                )
            elif self.state == "IN_TRADE":
                # Safety fallback (normal IN_TRADE handled by self.in_trade flag above).
                self.state = "IN_TRADE" if self.in_trade else "IDLE"

        aggression_spike_flag = int(aggression_spike)

        # Update short memory after decision so current row is available next step.
        self.pressure_history.append(pressure_state)
        self.absorption_history.append(absorption_state)
        self.prev_pressure_state = pressure_state
        self.prev_absorption_state = absorption_state
        self.prev_price = price
        self.prev_pressure_value = pressure_value

        # Update rolling histories after threshold checks (causal).
        self.speed_history.append(speed)
        self.abs_pressure_history.append(abs(pressure_value))

        out_state = self.state if self.state != "IN_TRADE" else "IN_TRADE"
        out_resolution = self.resolution_type if self.resolution_type else "none"
        out_confirmed_direction = (
            self.confirmed_direction if self.confirmed_direction in {"up", "down"} else "none"
        )
        out_in_trade = int(self.in_trade)

        if entry_signal == 0:
            entry_direction = "none"

        # If we just exited, ensure state cycle is fresh.
        if exit_signal == 1 and not self.in_trade:
            out_state = "IDLE"
            out_resolution = "none"
            out_confirmed_direction = "none"

        return {
            "state": out_state,
            "context_valid": context_valid,
            "resolution_type": out_resolution,
            "confirmed_direction": out_confirmed_direction,
            "aggression_spike_flag": aggression_spike_flag,
            "entry_signal": int(entry_signal),
            "entry_direction": entry_direction,
            "in_trade": out_in_trade,
            "exit_signal": int(exit_signal),
        }


def build_signal_stream(
    df: pd.DataFrame,
    window_size: int,
    max_trade_events: int,
) -> pd.DataFrame:
    machine = RealTimeStateMachine(window_size=window_size, max_trade_events=max_trade_events)
    outputs = []
    for _, row in df.iterrows():
        outputs.append(machine.process_event(row))
    return pd.DataFrame(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build real-time state-machine signal stream from event features."
    )
    parser.add_argument(
        "--input",
        default="event_features_with_regime_direction_location.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        default="signal_stream.csv",
        help="Output signal stream CSV path",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=100,
        help="Rolling window size for aggression thresholds (default: 100)",
    )
    parser.add_argument(
        "--max-trade-events",
        type=int,
        default=80,
        help="Time-based exit fallback in number of events (default: 80)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    use = df.dropna(subset=REQUIRED_COLUMNS).copy().reset_index(drop=True)
    stream = build_signal_stream(
        use,
        window_size=args.window_size,
        max_trade_events=args.max_trade_events,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stream.to_csv(output_path, index=False)

    print("=== Real-Time State Machine Build Summary ===")
    print(f"rows_input              : {len(df)}")
    print(f"rows_used               : {len(use)}")
    print(f"rows_output             : {len(stream)}")
    print("state_counts            :", stream["state"].value_counts(dropna=False).to_dict())
    print("resolution_counts       :", stream["resolution_type"].value_counts(dropna=False).to_dict())
    print(f"entry_signal_count      : {int(stream['entry_signal'].sum())}")
    print(f"exit_signal_count       : {int(stream['exit_signal'].sum())}")
    print(f"aggression_spike_count  : {int(stream['aggression_spike_flag'].sum())}")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
