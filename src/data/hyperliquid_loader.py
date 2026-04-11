import pandas as pd
from hyperliquid.info import Info
from datetime import datetime


class HyperliquidDataLoader:
    """
    Data loader for Hyperliquid exchange API
    Fetches OHLCV historical data and converts to standardized pandas DataFrame
    """

    def __init__(self):
        """Initialize Hyperliquid info client"""
        # Keep websocket disabled for simple data fetching workflows.
        # This aligns with documented test patterns using skip_ws=True.
        self.info = Info(base_url="https://api.hyperliquid.xyz", skip_ws=True)

    @staticmethod
    def _interval_to_milliseconds(interval: str) -> int:
        """Convert Hyperliquid interval string (e.g. 1m, 1h, 1d) to milliseconds."""
        unit = interval[-1]
        value = int(interval[:-1])
        if unit == "m":
            return value * 60_000
        if unit == "h":
            return value * 3_600_000
        if unit == "d":
            return value * 86_400_000
        raise ValueError(f"Unsupported interval format: {interval}")

    def fetch_ohlcv(self, symbol: str = "BTC", interval: str = "1m", limit: int = 500) -> pd.DataFrame:
        """
        Fetch OHLCV data from Hyperliquid

        Args:
            symbol: Asset symbol (BTC, ETH, etc.)
            interval: Timeframe interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to fetch (max 5000)

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
            DateTime index in UTC timezone
        """
        try:
            # Hyperliquid Info client exposes candles_snapshot(name=..., interval=..., startTime=..., endTime=...)
            # We derive a time window from `limit` and `interval`.
            interval_ms = self._interval_to_milliseconds(interval)
            end_time = int(pd.Timestamp.utcnow().timestamp() * 1000)
            start_time = end_time - (max(limit, 1) * interval_ms)

            candles = self.info.candles_snapshot(
                name=symbol,
                interval=interval,
                startTime=start_time,
                endTime=end_time,
            )

            if not candles:
                return pd.DataFrame()

            df = pd.DataFrame(candles)

            # Map fields (Hyperliquid format)
            df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)

            df = df.rename(columns={
                "o": "Open",
                "h": "High",
                "l": "Low",
                "c": "Close",
                "v": "Volume"
            })

            df = df.set_index("timestamp")

            df = df[["Open", "High", "Low", "Close", "Volume"]]

            # Ensure numeric types for analytics/backtesting
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Drop malformed rows if any
            df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

            # Apply limit manually
            if limit:
                df = df.tail(limit)

            return df.sort_index()

        except Exception as e:
            print(f"Error fetching Hyperliquid data: {str(e)}")
            return pd.DataFrame()