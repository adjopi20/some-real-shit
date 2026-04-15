from typing import Dict, Optional, ClassVar
import numbers


class CandleAggregator:
    TIMEFRAME_MS: ClassVar[Dict[str, int]] = {
        "1s": 1000,
        "5s": 5000,
        "15s": 15000,
        "1m": 60 * 1000,
        "3m": 3 * 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "2h": 2 * 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "6h": 6 * 60 * 60 * 1000,
        "8h": 8 * 60 * 60 * 1000,
        "12h": 12 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }

    __slots__ = (
        "timeframe",
        "interval_ms",
        "_current_candle",
        "_current_bucket",
        "output_precision",
        # optional duplicate protection
        # "_last_trade_id",
    )

    def __init__(self, timeframe: str = "1m", output_precision: int = 6):
        if timeframe not in self.TIMEFRAME_MS:
            supported = ", ".join(sorted(self.TIMEFRAME_MS.keys()))
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Supported: {supported}")

        self.timeframe = timeframe
        self.interval_ms = self.TIMEFRAME_MS[timeframe]
        self.output_precision = output_precision

        self._current_candle: Optional[Dict] = None
        self._current_bucket: Optional[int] = None

        # self._last_trade_id = None  # enable if your feed provides IDs

    # =========================
    # INTERNAL
    # =========================
    def _get_candle_bucket(self, timestamp_ms: int) -> int:
        return timestamp_ms - (timestamp_ms % self.interval_ms)

    def _round_output(self, output: Dict) -> Dict:
        return {
            k: round(v, self.output_precision)
            if isinstance(v, float)
            else v
            for k, v in output.items()
        }

    # =========================
    # MAIN
    # =========================
    def process_event(self, event: Dict) -> Optional[Dict]:
        # -------------------------
        # Input validation
        # -------------------------
        try:
            timestamp = int(event["timestamp"])
            price = float(event["price"])
            quantity = float(event["quantity"])
            side = event["side"]
        except (KeyError, TypeError, ValueError):
            return None

        # -------------------------
        # Data integrity check
        # -------------------------
        if price <= 0 or quantity <= 0:
            return None

        # -------------------------
        # Side validation
        # -------------------------
        if side not in ("buy", "sell"):
            return None

        # -------------------------
        # Optional duplicate protection
        # -------------------------
        # trade_id = event.get("trade_id")
        # if trade_id is not None:
        #     if self._last_trade_id is not None and trade_id <= self._last_trade_id:
        #         return None
        #     self._last_trade_id = trade_id

        event_bucket = self._get_candle_bucket(timestamp)

        # -------------------------
        # First event
        # -------------------------
        if self._current_candle is None:
            self._init_new_candle(event_bucket, price)
            self._update_candle(price, quantity, side)
            return None

        # -------------------------
        # Out-of-order protection (CRITICAL)
        # -------------------------
        if event_bucket < self._current_bucket:
            return None

        # -------------------------
        # New candle
        # -------------------------
        if event_bucket > self._current_bucket:
            completed = self._finalize_candle()

            self._init_new_candle(event_bucket, price)
            self._update_candle(price, quantity, side)

            return completed

        # -------------------------
        # Same candle
        # -------------------------
        self._update_candle(price, quantity, side)
        return None

    # =========================
    # CANDLE OPS
    # =========================
    def _init_new_candle(self, bucket_start: int, open_price: float):
        self._current_bucket = bucket_start
        self._current_candle = {
            "start_time": bucket_start,
            "end_time": bucket_start + self.interval_ms - 1,
            "open": open_price,
            "high": open_price,
            "low": open_price,
            "close": open_price,
            "volume": 0.0,
            "buy_volume": 0.0,
            "sell_volume": 0.0,
            "delta": 0.0,
            "trade_count": 0,
            "vwap_num": 0.0,  # sum(price * qty)
        }

    def _update_candle(self, price: float, quantity: float, side: str):
        c = self._current_candle

        # OHLC
        if price > c["high"]:
            c["high"] = price
        if price < c["low"]:
            c["low"] = price
        c["close"] = price

        # Volume
        c["volume"] += quantity
        c["trade_count"] += 1
        c["vwap_num"] += price * quantity

        # Flow
        if side == "buy":
            c["buy_volume"] += quantity
            c["delta"] += quantity
        else:
            c["sell_volume"] += quantity
            c["delta"] -= quantity

    def _finalize_candle(self) -> Dict:
        c = dict(self._current_candle)

        # VWAP
        c["vwap"] = (
            c["vwap_num"] / c["volume"]
            if c["volume"] > 0 else 0.0
        )
        del c["vwap_num"]

        # Close time (IMPORTANT)
        c["close_time"] = self._current_bucket + self.interval_ms

        return self._round_output(c)

    # =========================
    # UTILS
    # =========================
    def get_current_candle(self) -> Optional[Dict]:
        if self._current_candle is None:
            return None

        c = dict(self._current_candle)

        c["vwap"] = (
            c["vwap_num"] / c["volume"]
            if c["volume"] > 0 else 0.0
        )
        del c["vwap_num"]

        return self._round_output(c)
    
    def reset(self):
        self._current_candle = None
        self._current_bucket = None
        # self._last_trade_id = None