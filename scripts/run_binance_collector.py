import asyncio
import sys
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.binance_loader import BinanceDataLoader
from src.recorder.data_recorder import DataRecorder
from src.features.orderflow_engine import OrderFlowEngine
from src.features.candle_aggregator import CandleAggregator
from src.core.dispatcher import Dispatcher



async def main():
    filename = f"data/enriched_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"


    loader = BinanceDataLoader()
    recorder = DataRecorder(output_path=filename)
    orderflow_engine = OrderFlowEngine(window_type="time", window_size=5)
    candle_aggregator = CandleAggregator(timeframe="1m")
    
    dispatcher = Dispatcher(
        orderflow_engine=orderflow_engine,
        candle_aggregator=candle_aggregator,
        recorder=recorder
    )

    async def handle_event(event: dict):
        dispatcher.process(event)
        print(f"Processed event @ {event['timestamp']}")

    async def handle_reconnect():
        orderflow_engine.reset_session()
        candle_aggregator.reset()

    # IMPORTANT:
    # On WebSocket reconnect, call:
    # orderflow_engine.reset_session()
    
    await loader.stream_trades(
        handle_event, 
        on_reconnect=handle_reconnect)

if __name__ == "__main__":
    asyncio.run(main())