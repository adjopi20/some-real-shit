import json
from pathlib import Path
from typing import Dict


class DataRecorder:
    """Append incoming events to a JSONL file."""

    def __init__(self, output_path: str = "data/binance_trades.jsonl"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: Dict) -> None:
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")