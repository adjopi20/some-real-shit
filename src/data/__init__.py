"""
Data Loader Module for fetching and preparing market data
"""
import yfinance as yf
import pandas as pd
from typing import Optional, Tuple
from datetime import datetime


class DataLoader:
    """
    Handles fetching, cleaning and preparing historical market data
    """
    
    @staticmethod
    def fetch_data(
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
        interval: str = "1h"
    ) -> pd.DataFrame:
        """
        Fetch historical price data from Yahoo Finance
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (defaults to today)
            interval: Data interval (1d, 1h, 15m, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
            
        # Use period if provided, otherwise use start/end dates
        if period is not None:
            data = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                progress=False
            )
        else:
            data = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False
            )
        
        # Clean and standardize dataframe
        data = DataLoader._clean_data(data)
        return data
    
    @staticmethod
    def _clean_data(data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare raw price data"""
        # Drop rows with missing values
        data = data.dropna()
        
        # Flatten multi-index columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # Ensure correct column order
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return data[required_columns]
    
    @staticmethod
    def calculate_returns(data: pd.DataFrame, column: str = 'Close') -> pd.Series:
        """Calculate log returns for a price series"""
        return pd.Series(
            data[column].pct_change(),
            name='returns'
        )
    
    @staticmethod
    def to_lowercase_columns(data: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to lowercase format expected by strategies"""
        data.columns = data.columns.str.lower()
        return data
