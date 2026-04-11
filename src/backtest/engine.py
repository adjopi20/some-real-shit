import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

from ..strategy.fabio_strategy import BaseStrategy


@dataclass
class BacktestResult:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    equity_curve: pd.Series
    trades: List[Dict]


class BacktestEngine:
    """
    Backtest Engine for strategy execution and performance calculation
    """
    
    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.001, slippage: float = 0.0005):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        self.equity = initial_capital
        self.position = 0
        self.entry_price = 0.0
        
        self.equity_history = []
        self.trades = []
        
    def run(self, data: pd.DataFrame, strategy: BaseStrategy) -> BacktestResult:
        """
        Run backtest on historical data
        
        Args:
            data: OHLCV DataFrame with standard columns
            strategy: Strategy instance implementing BaseStrategy interface
            
        Returns:
            BacktestResult object with performance metrics
        """
        if data.empty:
            raise ValueError("Empty data provided for backtest")
            
        # Reset state
        self.equity = self.initial_capital
        self.position = 0
        self.equity_history = []
        self.trades = []
        
        # Initialize strategy
        strategy.init()
        
        # Iterate over each bar
        for idx, row in data.iterrows():
            # Update strategy with current bar
            signal = strategy.next(row)
            
            # Execute trading signals
            self._execute_signal(row, signal)
            
            # Record equity
            current_value = self.equity + (self.position * row['close'])
            self.equity_history.append(current_value)
            
        # Create equity curve
        equity_curve = pd.Series(self.equity_history, index=data.index)
        
        # Calculate performance metrics
        result = self._calculate_metrics(equity_curve)
        
        return result
    
    def _execute_signal(self, bar: pd.Series, signal: int):
        """
        Execute trading signal: -1 = SELL, 0 = HOLD, 1 = BUY
        """
        price = bar['close']
        
        # Apply slippage
        if signal == 1:
            execution_price = price * (1 + self.slippage)
        elif signal == -1:
            execution_price = price * (1 - self.slippage)
        else:
            return
            
        if signal == 1 and self.position <= 0:
            # Open long position
            if self.position < 0:
                # Close existing short first
                self.equity += self.position * execution_price * (1 - self.commission)
                self._record_trade('CLOSE_SHORT', execution_price, bar.name)
                self.position = 0
                
            # Open new long
            self.position = self.equity / execution_price
            self.equity -= self.position * execution_price * (1 + self.commission)
            self.entry_price = execution_price
            self._record_trade('BUY', execution_price, bar.name)
            
        elif signal == -1 and self.position >= 0:
            # Open short position
            if self.position > 0:
                # Close existing long first
                self.equity += self.position * execution_price * (1 - self.commission)
                self._record_trade('CLOSE_LONG', execution_price, bar.name)
                self.position = 0
                
            # Open new short
            self.position = -(self.equity / execution_price)
            self.equity += abs(self.position) * execution_price * (1 - self.commission)
            self.entry_price = execution_price
            self._record_trade('SELL', execution_price, bar.name)
    
    def _record_trade(self, action: str, price: float, timestamp: datetime):
        self.trades.append({
            'action': action,
            'price': price,
            'timestamp': timestamp,
            'equity': self.equity
        })
    
    def _calculate_metrics(self, equity_curve: pd.Series) -> BacktestResult:
        """Calculate all performance metrics"""
        total_return = (equity_curve.iloc[-1] / self.initial_capital) - 1
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        returns = equity_curve.pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = np.sqrt(252 * 390) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0.0
            
        # Max drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        wins = sum(1 for i in range(1, len(equity_curve)) if equity_curve.iloc[i] > equity_curve.iloc[i-1])
        win_rate = wins / len(equity_curve) if len(equity_curve) > 0 else 0.0
        
        total_trades = len(self.trades)
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            equity_curve=equity_curve,
            trades=self.trades
        )