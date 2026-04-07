#!/usr/bin/env python3
"""
Example backtest execution script
Demonstrates complete backtest workflow
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import DataLoader
from src.strategy import Strategy
from src.backtest import BacktestEngine
from src.metrics import calculate_all_metrics


def main():
    print("=== Algorithmic Trading Backtest System ===")
    print()
    
    # 1. Load historical market data
    print("✅ Loading historical data...")
    data_loader = DataLoader()
    data = data_loader.fetch_data(
        ticker="TSLA",
        # start_date="2024-01-01",
        # end_date="2025-01-01",
        period="1y",
        interval="1h"
    )
    print(f"   Loaded {len(data)} trading days for TSLA")
    print()
    
    # 2. Initialize strategy
    print("✅ Initializing strategy...")
    # Create concrete strategy implementation
    class MovingAverageCrossoverStrategy(Strategy):
        def generate_signals(self, data: pd.DataFrame) -> pd.Series:
            short_window = self.parameters.get('short_window', 50)
            long_window = self.parameters.get('long_window', 200)
            
            # Calculate moving averages
            short_ma = data['Close'].rolling(window=short_window).mean()
            long_ma = data['Close'].rolling(window=long_window).mean()
            
            # Generate signals: 1 = LONG, -1 = SHORT, 0 = HOLD
            signals = pd.Series(0, index=data.index)
            signals[short_ma > long_ma] = 1
            signals[short_ma <= long_ma] = 0
            
            self.signals = signals
            self.positions = signals
            
            return signals
    
    strategy = MovingAverageCrossoverStrategy({
        'short_window': 9,
        'long_window': 20
    })
    print(f"   SMA Crossover Strategy (9/20)")
    print()
    
    # 3. Run backtest
    print("✅ Executing backtest...")
    engine = BacktestEngine(
        initial_capital=100000.0,
        commission=0.001
    )
    
    result = engine.run(data, strategy)
    print(f"   Backtest completed successfully")
    print()
    
    # 4. Display results
    print("=== BACKTEST RESULTS ===")
    print(f"Total Return:      {result.total_return:.2%}")
    print(f"Sharpe Ratio:      {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown:      {result.max_drawdown:.2%}")
    print(f"Win Rate:          {result.win_rate:.2%}")
    print(f"Total Trades:      {result.total_trades}")
    print()
    
    # 5. Calculate extended metrics
    print("=== EXTENDED METRICS ===")
    metrics = calculate_all_metrics(result.equity_curve)
    for name, value in metrics.items():
        if isinstance(value, float):
            if abs(value) < 10:
                print(f"{name.replace('_', ' ').title()}:  {value:.3f}")
            else:
                print(f"{name.replace('_', ' ').title()}:  {value:.0f}")
    print()
    
    # 6. Generate equity curve plot
    print("✅ Generating performance chart...")
    plt.figure(figsize=(12, 6))
    result.equity_curve.plot(label='Strategy Equity', linewidth=2)
    plt.title('Backtest Equity Curve')
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('equity_curve.png', dpi=150, bbox_inches='tight')
    print("   Chart saved as 'equity_curve.png'")
    print()
    
    print("✅ Backtest completed successfully!")


if __name__ == "__main__":
    main()