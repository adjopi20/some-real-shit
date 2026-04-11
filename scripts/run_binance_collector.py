import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.binance_loader import BinanceDataLoader
from src.recorder.data_recorder import DataRecorder
from src.features.orderflow_engine import OrderFlowEngine


async def main():
    loader = BinanceDataLoader()
    recorder = DataRecorder()
    feature_engine = OrderFlowEngine(window_type="time", window_size=5)

    async def handle_event(event: dict):
        enriched_event = feature_engine.process_event(event)
        print(enriched_event)
        recorder.record(enriched_event)

    await loader.stream_trades(handle_event)


if __name__ == "__main__":
    asyncio.run(main())