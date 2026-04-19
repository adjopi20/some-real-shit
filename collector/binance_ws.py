"""Minimal Binance websocket trade stream collector."""

from __future__ import annotations

import asyncio
import json
import time
from inspect import isawaitable
from typing import Callable, Optional

import websockets

from collector import config


TradeCallback = Callable[[tuple[int, int, int, float, float, int]], object]
LogCallback = Optional[Callable[[str], None]]


class BinanceTradeStream:
    """Streams raw Binance trade events in minimal tuple format.

    Internal event tuple layout:
        (timestamp_ms, recv_ts_ns, local_id, price, quantity, side)

    Deterministic ordering rule:
        final ordering key = (timestamp_ms, local_id)
    """

    def __init__(self, ws_url: str = config.WS_URL, gap_threshold_ms: int = config.GAP_THRESHOLD_MS) -> None:
        self.ws_url = ws_url
        self.gap_threshold_ms = gap_threshold_ms
        self._local_id = 0
        self._last_timestamp_ms = 0
        self._gap_count = 0
        self._last_gap_logged_count = 0
        self._invalid_count = 0
        self._total_events = 0

    def _parse_message(self, message: dict) -> tuple[int, int, int, float, float, int] | None:
        """Parse raw Binance payload into minimal tuple; returns None for invalid trades."""
        try:
            timestamp_ms = int(message["T"])
            price = float(message["p"])
            quantity = float(message["q"])
            side = -1 if bool(message["m"]) else 1
        except (KeyError, TypeError, ValueError):
            return None

        # Data integrity safeguard: validate before buffering/writing.
        if timestamp_ms <= 0 or price <= 0.0 or quantity <= 0.0:
            self._invalid_count += 1
            return None

        self._local_id += 1
        recv_ts_ns = time.time_ns()

        if self._last_timestamp_ms and (timestamp_ms - self._last_timestamp_ms) > self.gap_threshold_ms:
            self._gap_count += 1
        self._last_timestamp_ms = timestamp_ms

        return (timestamp_ms, recv_ts_ns, self._local_id, price, quantity, side)

    async def stream(
        self,
        on_event: TradeCallback,
        *,
        logger: LogCallback = None,
        reconnect_delay_sec: float = 2.0,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """Continuously stream trades with automatic reconnects."""
        while True:
            if stop_event is not None and stop_event.is_set():
                return

            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20, close_timeout=5) as ws:
                    if logger:
                        logger(f"connected: {self.ws_url}")

                    async for raw in ws:
                        if stop_event is not None and stop_event.is_set():
                            return

                        try:
                            message = json.loads(raw)
                        except json.JSONDecodeError:
                            self._invalid_count += 1
                            continue

                        event = self._parse_message(message)
                        if event is None:
                            continue

                        self._total_events += 1
                        result = on_event(event)
                        if isawaitable(result):
                            await result

                        if (
                            logger
                            and self._gap_count
                            and self._gap_count % config.GAP_LOG_EVERY == 0
                            and self._gap_count != self._last_gap_logged_count
                        ):
                            self._last_gap_logged_count = self._gap_count
                            logger(f"GAP DETECTED count={self._gap_count} last_ts={self._last_timestamp_ms}")

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if logger:
                    logger(f"websocket disconnected ({exc}); reconnecting in {reconnect_delay_sec:.1f}s")
                await asyncio.sleep(reconnect_delay_sec)
