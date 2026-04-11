from typing import Optional
import pandas as pd
from .hyperliquid_loader import HyperliquidDataLoader


class DataLoader:
    """
    Unified data loader interface for multiple data sources
    Abstracts away specific exchange implementations
    """
    
    @staticmethod
    def fetch_data(source: str, symbol: str, interval: str = "1m", limit: int = 500) -> pd.DataFrame:
        """
        Fetch OHLCV data from specified data source
        
        Args:
            source: Data source identifier ('hyperliquid' currently supported)
            symbol: Trading pair symbol (BTC, ETH, etc.)
            interval: Timeframe interval (1m, 5m, 15m, 1h, etc.)
            limit: Number of candles to retrieve
            
        Returns:
            Standardized pandas DataFrame with OHLCV data
            
        Raises:
            ValueError: If unsupported data source is requested
        """
        if source.lower() == "hyperliquid":
            loader = HyperliquidDataLoader()
            return loader.fetch_ohlcv(symbol, interval, limit)
        else:
            raise ValueError(f"Unsupported data source: {source}. Currently only 'hyperliquid' is supported.")