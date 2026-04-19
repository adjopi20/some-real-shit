"""Collector runtime configuration."""

SYMBOL = "btcusdt"
WS_URL = "wss://fstream.binance.com/ws/btcusdt@trade"

OUTPUT_DIR = "storage/btcusdt/"
CHUNK_SIZE = 50_000
FLUSH_INTERVAL_SEC = 5

# Gap detection hook (logging only)
GAP_THRESHOLD_MS = 1_000
GAP_LOG_EVERY = 100

# Aggregated logging cadence
STATS_LOG_INTERVAL_SEC = 10
