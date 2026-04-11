"""
Example Backtest for Fabio Valentini Pro Scalper Strategy
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from src.fabio_valentini_strategy import FabioValentiniProScalper
from src.backtest import Backtest
from src.metrics import calculate_metrics


def main():
    print("⚡ Fabio Valentini Pro Scalper Backtest")
    print("=" * 50)

    # Load historical data (1min intervals for scalping)
    print("\nLoading market data...")
    ticker = "QQQ"  # Nasdaq 100 - perfect for this scalping strategy
    
    # Download intraday data
    data = yf.download(
        tickers=ticker,
        start="2025-01-01",
        end="2025-03-01",
        interval="1m",
        auto_adjust=True
    )
    
    # Format dataframe
    data = data.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    data['datetime'] = data.index
    print(f"Loaded {len(data)} bars | {ticker} 1min data")

    # Initialize strategy
    print("\nInitializing strategy...")
    strategy = FabioValentiniProScalper(parameters={
        'vp_length': 50,
        'absorb_vol_mult': 2.0,
        'rr_ratio': 2.0,
        'max_daily_losses': 3,
        'orb_minutes': 30
    })

    # Generate signals
    print("Generating trading signals...")
    signals = strategy.generate_signals(data)

    # Run backtest
    print("\nRunning backtest...")
    backtest = Backtest(
        initial_capital=100000,
        commission=0.001,  # 0.1% commission per trade
        slippage=0.0005     # 0.05% slippage
    )
    
    results = backtest.run(data, signals, strategy)
    
    # Calculate performance metrics
    metrics = calculate_metrics(results)
    
    # Print report
    print("\n📊 BACKTEST RESULTS")
    print("=" * 50)
    print(f"Total Return:          {metrics['total_return_pct']:.2f}%")
    print(f"Win Rate:              {metrics['win_rate_pct']:.2f}%")
    print(f"Profit Factor:         {metrics['profit_factor']:.2f}")
    print(f"Max Drawdown:          {metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio:          {metrics['sharpe_ratio']:.2f}")
    print(f"Total Trades:          {metrics['total_trades']}")
    print(f"Average R:R:           {metrics['avg_rr']:.2f}")
    print(f"Winning Trades:        {metrics['winning_trades']}")
    print(f"Losing Trades:         {metrics['losing_trades']}")
    print("\n✅ Strategy converted and ready for backtesting!")
    
    # Save results
    results.to_csv('fabio_backtest_results.csv')
    print("\nResults saved to fabio_backtest_results.csv")


if __name__ == "__main__":
    main()