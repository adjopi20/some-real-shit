from collections import deque
from typing import Deque, Tuple, Dict, Optional, Literal
import numbers

Event = Tuple[int, float, float, Literal["buy", "sell"]]


class OrderFlowEngine:
    def __init__(
        self,
        window_type: Literal["time", "count"] = "time",
        window_size: int = 5,
        absorption_volume_threshold: float = 3.5,
        absorption_price_threshold: float = 0.0006,
        baseline_alpha: float = 0.3,
        output_precision: int = 8
    ):
        self.window_type = window_type
        self.window_size = window_size
        self.absorption_volume_threshold = absorption_volume_threshold
        self.absorption_price_threshold = absorption_price_threshold
        self.baseline_alpha = baseline_alpha

        self._events: Deque[Event] = deque()

        self._buy_volume = 0.0
        self._sell_volume = 0.0
        self._window_cum_delta = 0.0
        self._session_cum_delta = 0.0

        self._min_price: Optional[float] = None
        self._max_price: Optional[float] = None

        # Baseline
        self._baseline_volume = 0.0
        self._last_baseline_update_ts = None
        self._baseline_max_age_ms = window_size * 1000 * 3
        self._baseline_update_count = 0

        # Data health
        self._last_event_ts = None
        self._last_event = None
        self._stale_threshold = 10_000  # ms

        self.output_precision = output_precision

    # =========================
    # WINDOW MANAGEMENT
    # =========================
    def _clean_expired_events(self, now_ts: int):
        removed = []

        if self.window_type == "time":
            cutoff = now_ts - self.window_size * 1000
            while self._events and self._events[0][0] < cutoff:
                removed.append(self._events.popleft())
        else:
            while len(self._events) > self.window_size:
                removed.append(self._events.popleft())

        need_refresh = False

        for _, price, qty, side in removed:
            if side == "buy":
                self._buy_volume -= qty
                self._window_cum_delta -= qty
            else:
                self._sell_volume -= qty
                self._window_cum_delta += qty

            if price == self._min_price or price == self._max_price:
                need_refresh = True

        if not self._events:
            self._min_price = None
            self._max_price = None
        elif need_refresh:
            self._min_price = min(e[1] for e in self._events)
            self._max_price = max(e[1] for e in self._events)

    def _update_price_extremes(self, price: float):
        if self._min_price is None or price < self._min_price:
            self._min_price = price
        if self._max_price is None or price > self._max_price:
            self._max_price = price

    # =========================
    # BASELINE (EMA)
    # =========================
    def _update_baseline(self, now_ts: int, current_volume: float):
        if self.window_type == "time":
            if self._last_baseline_update_ts is None:
                self._last_baseline_update_ts = now_ts
                self._baseline_volume = current_volume
                self._baseline_update_count += 1
                return

            if now_ts - self._last_baseline_update_ts < self.window_size * 1000:
                return

            self._last_baseline_update_ts = now_ts

        else:
            # ✅ FIX: ensure timestamp exists in count mode
            if self._last_baseline_update_ts is None:
                self._last_baseline_update_ts = now_ts

        if self._baseline_volume == 0:
            self._baseline_volume = current_volume
        else:
            a = self.baseline_alpha
            self._baseline_volume = a * current_volume + (1 - a) * self._baseline_volume
        self._baseline_update_count += 1
    
    # =========================
    # ABSORPTION
    # =========================
    def _detect_absorption(self, total_volume: float, now_ts: int):
        """
        Detects high-volume, low-price-movement conditions.

        Interpretation:
        - Indicates passive liquidity absorbing aggressive orders.
        - This is NOT a directional signal.

        Use cases:
        - Volatility expansion precursor
        - Exhaustion detection
        - Market regime classification

        Do NOT use this alone to predict price direction.
        """
        if len(self._events) < 5 or self._baseline_volume == 0:
            return False

        if self._last_baseline_update_ts is None:
            return False

        # NEW: freshness check (critical)
        if now_ts - self._last_baseline_update_ts > self._baseline_max_age_ms:
            return False

        if self._min_price is None or self._max_price is None:
            return False

        price_range = self._max_price - self._min_price
        ref_price = self._max_price if self._max_price else 1.0

        return (
            total_volume > self._baseline_volume * self.absorption_volume_threshold
            and (price_range / ref_price) < self.absorption_price_threshold
        )
    
    def _round_output(self, output: Dict) -> Dict:
        return {
            k: round(float(v), self.output_precision) 
            if isinstance(v, numbers.Real) and not isinstance(v, bool) 
            else v
            for k, v in output.items()
        }

    # =========================
    # MAIN
    # =========================
    def process_event(self, event: Dict) -> Optional[Dict]:
        try:
            ts = int(event["timestamp"])
            price = float(event["price"])
            qty = float(event["quantity"])
            side = event["side"]
        except (KeyError, TypeError, ValueError):
            return None

        if self._last_event_ts and ts < self._last_event_ts:
            return None

        current_event = (ts, price, qty, side)
        if self._last_event == current_event:
            return None
        self._last_event = current_event

        # Data integrity check
        if price <= 0 or qty <= 0:
            return None

        # Stale protection
        if self._last_event_ts and (ts - self._last_event_ts > self._stale_threshold):
            self.reset_window()

        self._last_event_ts = ts

        # Clean window
        self._clean_expired_events(ts)

        # Delta
        delta = qty if side == "buy" else -qty
        self._session_cum_delta += delta
        self._window_cum_delta += delta

        # Volume
        if side == "buy":
            self._buy_volume += qty
        else:
            self._sell_volume += qty

        # Store
        self._events.append((ts, price, qty, side))
        self._update_price_extremes(price)

        total = self._buy_volume + self._sell_volume
        trade_count = len(self._events)

        # Detect BEFORE baseline update
        absorption = self._detect_absorption(total, ts)

        # Update baseline AFTER
        self._update_baseline(ts, total)

        # =========================
        # METRICS
        # =========================
        volume_per_sec = 0.0
        delta_per_sec = 0.0
        trade_intensity = 0.0

        if trade_count > 1:
            time_span = (ts - self._events[0][0]) / 1000
            if time_span > 0.001:
                volume_per_sec = total / time_span
                delta_per_sec = self._window_cum_delta / time_span
                trade_intensity = trade_count / time_span
            else:
                volume_per_sec = 0.0
                delta_per_sec = 0.0
                trade_intensity = 0.0

        imbalance = (
            (self._buy_volume - self._sell_volume) / total
            if total > 0 else 0.0
        )

        # NEW: directional pressure (key signal component)
        pressure = (
            self._window_cum_delta / total
            if total > 0 else 0.0
        )

        price_range = (
            (self._max_price - self._min_price)
            if self._min_price is not None and self._max_price is not None
            else 0.0
        )

        warm = (
            len(self._events) >= 5 and
            self._baseline_update_count >= 3
        )

        return self._round_output({
            "timestamp": ts,
            "price": price,
            "delta": delta,
            "window_cum_delta": self._window_cum_delta,
            "session_cum_delta": self._session_cum_delta,
            "buy_volume": self._buy_volume,
            "sell_volume": self._sell_volume,
            "pressure": pressure,  # NEW
            "volume_per_sec": volume_per_sec,
            "delta_per_sec": delta_per_sec,
            "trade_intensity": trade_intensity,
            "price_range": price_range,
            "absorption": absorption,
            "warm": warm,
        })

    # =========================
    # RESET
    # =========================
    def reset_window(self):
        self._events.clear()
        self._buy_volume = 0.0
        self._sell_volume = 0.0
        self._window_cum_delta = 0.0
        self._min_price = None
        self._max_price = None
        self._last_event_ts = None  # ✅ FIX

    def reset_session(self):
        self.reset_window()
        self._baseline_volume = 0.0
        self._last_baseline_update_ts = None
        self._baseline_update_count = 0
        self._session_cum_delta = 0.0
        self._last_event = None