"""
Base Strategy Module - Abstract base class for all trading strategies
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Dict


class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    All custom strategies must inherit from this class and implement
    the generate_signals method.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize strategy with optional parameters
        
        Args:
            parameters: Dictionary of strategy parameters
        """
        self.parameters = parameters or {}
        self.positions = None
        self.signals = None
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from market data
        
        Args:
            data: DataFrame with OHLCV market data
            
        Returns:
            Series of trading signals (-1 = SHORT, 0 = HOLD, 1 = LONG)
        """
        pass
    
    def get_positions(self) -> pd.Series:
        """
        Get current position allocations
        
        Returns:
            Series of position sizes
        """
        return self.positions
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return self.parameters.copy()