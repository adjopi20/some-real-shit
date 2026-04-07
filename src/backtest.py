"""
Backtest Engine Module - Executes strategy simulation on historical data
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from dataclasses import dataclass

from .strategy import Strategy


@dataclass
class BacktestResult:
    """Container for backtest execution results"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    equity_curve: pd.Series
    positions: pd.Series
    signals: pd.Series


class BacktestEngine:
    """
    Backtesting engine that runs strategies against historical data
    and calculates performance metrics
    """
    
    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.001):
        """
        Initialize backtest engine
        
        Args:
            initial_capital: Starting capital amount
            commission: Trading commission fee (per trade value)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        
    def run(self, data: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        """
        Execute backtest for given data and strategy
        
        Args:
            data: OHLCV market data
            strategy: Trading strategy implementation
            
        Returns:
            BacktestResult object with performance metrics
        """
        # Generate trading signals
        signals = strategy.generate_signals(data)
        
        # Calculate positions and returns
        positions = signals.shift(1)  # Enter position day after signal
        returns = data['Close'].pct_change()
        
        # Apply commission costs
        trade_changes = positions.diff().abs()
        costs = trade_changes * self.commission
        
        # Calculate strategy returns
        strategy_returns = (positions * returns) - costs
        
        # Build equity curve
        equity_curve = (1 + strategy_returns).cumprod() * self.initial_capital
        equity_curve.iloc[0] = self.initial_capital
        
        # Calculate performance metrics
        total_return = (equity_curve.iloc[-1] / self.initial_capital) - 1
        sharpe_ratio = self._calculate_sharpe(strategy_returns)
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        win_rate, total_trades = self._calculate_trade_stats(strategy_returns, positions)
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            equity_curve=equity_curve,
            positions=positions,
            signals=signals
        )
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sharpe Ratio"""
        excess_returns = returns - (risk_free_rate / 252)
        if excess_returns.std() == 0:
            return 0.0
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown from peak"""
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return drawdown.min()
    
    def _calculate_trade_stats(self, returns: pd.Series, positions: pd.Series) -> Tuple[float, int]:
        """Calculate win rate and total number of trades"""
        trade_changes = positions.diff().abs() != 0
        total_trades = int(trade_changes.sum())
        
        if total_trades == 0:
            return 0.0, 0
            
        trade_returns = returns[trade_changes.shift(1).fillna(False)]
        winning_trades = (trade_returns > 0).sum()
        win_rate = winning_trades / len(trade_returns)
        
        return win_rate, total_trades