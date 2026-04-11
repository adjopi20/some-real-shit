import json
from typing import Awaitable, Callable, Dict

import websockets


class BinanceDataLoader:
    """Minimal Binance trade stream loader for BTCUSDT."""

    WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"

    @staticmethod
    def parse_trade(message: Dict) -> Dict:
        """Normalize Binance trade payload into common trade event format."""
        return {
            "source": "binance",
            "type": "trade",
            "timestamp": int(message["T"]),
            "price": float(message["p"]),
            "quantity": float(message["q"]),
            "side": "sell" if message.get("m", False) else "buy",
        }

    async def stream_trades(self, on_event: Callable[[Dict], Awaitable[None]]):
        """Connect and continuously stream normalized trade events."""
        async with websockets.connect(self.WS_URL) as websocket:
            async for raw in websocket:
                message = json.loads(raw)
                event = self.parse_trade(message)
                await on_event(event)