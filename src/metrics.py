"""
Performance Metrics Module - Additional performance calculation utilities
"""
import pandas as pd
import numpy as np
from typing import Dict, Any


def calculate_all_metrics(equity_curve: pd.Series) -> Dict[str, Any]:
    """
    Calculate complete set of performance metrics from equity curve
    
    Args:
        equity_curve: Series of equity values over time
        
    Returns:
        Dictionary with all calculated metrics
    """
    returns = equity_curve.pct_change().dropna()
    
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min()
    avg_drawdown = drawdown.mean()
    
    # Calculate risk adjusted returns
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
    sortino = np.sqrt(252) * returns.mean() / returns[returns < 0].std() if len(returns[returns < 0]) > 0 and returns[returns < 0].std() != 0 else 0
    calmar = returns.mean() * 252 / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Calculate trade statistics
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    
    win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
    profit_factor = abs(positive_returns.sum() / negative_returns.sum()) if negative_returns.sum() != 0 else float('inf')
    
    # Calculate volatility
    annualized_volatility = returns.std() * np.sqrt(252)
    
    return {
        'total_return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1,
        'annualized_return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(equity_curve)) - 1,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'average_drawdown': avg_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'annualized_volatility': annualized_volatility,
        'total_days': len(equity_curve)
    }