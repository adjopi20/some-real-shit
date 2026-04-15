from typing import Dict, Optional
from src.features.orderflow_engine import OrderFlowEngine
from src.features.candle_aggregator import CandleAggregator
from src.recorder.data_recorder import DataRecorder
import asyncio


class Dispatcher:
    """
    Central event dispatcher for real-time BTC order flow data collection.
    Routes trade events to processing engines and records combined output.
    """

    def __init__(
        self,
        orderflow_engine: OrderFlowEngine,
        candle_aggregator: CandleAggregator,
        recorder: DataRecorder
    ):
        self.orderflow_engine = orderflow_engine
        self.candle_aggregator = candle_aggregator
        self.recorder = recorder

    def process(self, event: Dict) -> None:
        """
        Process a single trade event.
        
        Args:
            event: Trade event dictionary with fields:
                timestamp (int ms), price (float), quantity (float), side ("buy"/"sell")
        """
        # Process event through both engines
        flow = self.orderflow_engine.process_event(event)
        candle = self.candle_aggregator.process_event(event)

        # Only log when flow is warmed up
        if flow and flow.get("warm", False):
            record = {
                "ts": event["timestamp"],
                "flow": flow,
                "candle": candle
            }
            asyncio.create_task(
                asyncio.to_thread(self.recorder.record, record)
            )
        
