import json
import asyncio
from typing import Awaitable, Callable, Dict, Optional

import websockets

class BinanceDataLoader:
    """Minimal Binance trade stream loader for BTCUSDT."""

    WS_URL = "wss://fstream.binance.com/ws/btcusdt@trade"

    def __init__(self):
        self._discard_count = 0

    @staticmethod
    def parse_trade(message: Dict) -> Optional[Dict]:
        """Normalize Binance trade payload into common trade event format."""
        try:
            price = float(message["p"])
            quantity = float(message["q"])

            # Validate price and quantity
            if price <= 0 or quantity <= 0:
                return None

            return {
                "source": "binance",
                "type": "trade",
                "timestamp": int(message["T"]),
                "price": price,
                "quantity": quantity,
                "side": "sell" if message.get("m", False) else "buy",
            }
        except (KeyError, TypeError, ValueError):
            return None

    async def stream_trades(
        self,
        on_event: Callable[[Dict], Awaitable[None]],
        on_reconnect: Optional[Callable[[], Awaitable[None]]] = None,
        logger: Optional[Callable[[str], None]] = None
    ):
        """Connect and continuously stream normalized trade events."""
        while True:
            try:
                async with websockets.connect(self.WS_URL) as websocket:
                    async for raw in websocket:
                        try:
                            message = json.loads(raw)
                            event = self.parse_trade(message)
                            if not event:
                                self._discard_count += 1
                                if logger and self._discard_count % 1000 == 0:
                                    logger(f"Discarded {self._discard_count} events")

                                continue

                            await on_event(event)

                        except Exception as e:
                            if logger:
                                logger(f"Processing error: {e} | raw={raw}")
                            continue

            except Exception as e:
                if logger:
                    logger(f"Connection dropped: {e}, reconnecting in 5s...")
                if on_reconnect:
                    if logger:
                        logger("Resetting session state after reconnect")
                    await on_reconnect()  # ✅ CRITICAL
                await asyncio.sleep(5)
