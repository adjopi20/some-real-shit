"""
Unit tests for CandleAggregator class.

Tests cover all requirements including edge cases, aggregation logic,
and candle closing behavior.
"""
import pytest
from src.features.candle_aggregator import CandleAggregator


def test_initialization():
    """Test that aggregator initializes correctly with valid timeframes."""
    agg = CandleAggregator("1m")
    assert agg.timeframe == "1m"
    assert agg.interval_ms == 60000
    assert agg.get_current_candle() is None
    
    with pytest.raises(ValueError):
        CandleAggregator("invalid_timeframe")


def test_single_candle_aggregation():
    """Test that multiple events in same bucket are aggregated correctly."""
    agg = CandleAggregator("1m")
    
    # First event
    event1 = {
        "timestamp": 1700000000000,  # Exact bucket start
        "price": 100.0,
        "quantity": 1.5,
        "side": "buy"
    }
    
    result = agg.process_event(event1)
    assert result is None
    
    current = agg.get_current_candle()
    assert current["open"] == 100.0
    assert current["high"] == 100.0
    assert current["low"] == 100.0
    assert current["close"] == 100.0
    assert current["volume"] == 1.5
    assert current["buy_volume"] == 1.5
    assert current["sell_volume"] == 0.0
    assert current["delta"] == 1.5
    assert current["trade_count"] == 1
    
    # Second event same bucket
    event2 = {
        "timestamp": 1700000030000,  # 30 seconds into same minute
        "price": 105.0,
        "quantity": 0.8,
        "side": "sell"
    }
    
    result = agg.process_event(event2)
    assert result is None
    
    current = agg.get_current_candle()
    assert current["high"] == 105.0
    assert current["low"] == 100.0
    assert current["close"] == 105.0
    assert current["volume"] == 2.3
    assert current["buy_volume"] == 1.5
    assert current["sell_volume"] == 0.8
    assert current["delta"] == 0.7  # 1.5 - 0.8
    assert current["trade_count"] == 2


def test_candle_close_and_new_candle():
    """Test that crossing bucket boundary returns completed candle."""
    agg = CandleAggregator("1m")
    
    # First bucket events - timestamp exactly aligned to minute boundary
    agg.process_event({
        "timestamp": 1700000040000,  # Exact minute start (divisible by 60000)
        "price": 100.0,
        "quantity": 1.0,
        "side": "buy"
    })
    
    # Event in NEXT bucket - this should trigger candle close
    completed = agg.process_event({
        "timestamp": 1700000100000,  # Next exact minute
        "price": 102.0,
        "quantity": 0.5,
        "side": "sell"
    })
    
    assert completed is not None
    assert completed["start_time"] == 1700000040000
    assert completed["end_time"] == 1700000099999
    assert completed["volume"] == 1.0
    assert completed["delta"] == 1.0
    assert completed["trade_count"] == 1
    
    # New candle is now active
    current = agg.get_current_candle()
    assert current["start_time"] == 1700000100000
    assert current["open"] == 102.0


def test_gap_between_candles():
    """Test handling gaps with no trades between candles."""
    agg = CandleAggregator("1m")
    
    # First minute
    agg.process_event({
        "timestamp": 1700000040000,
        "price": 100.0,
        "quantity": 1.0,
        "side": "buy"
    })
    
    # Jump 3 minutes ahead
    completed = agg.process_event({
        "timestamp": 1700000220000,
        "price": 105.0,
        "quantity": 2.0,
        "side": "buy"
    })
    
    # Only the first candle is returned - empty gaps are skipped
    assert completed is not None
    assert completed["start_time"] == 1700000040000
    
    current = agg.get_current_candle()
    assert current["start_time"] == 1700000220000


def test_ohlc_calculation():
    """Test correct OHLC values with multiple price movements."""
    agg = CandleAggregator("1m")
    
    events = [
        {"timestamp": 1700000000000, "price": 100.0, "quantity": 1, "side": "buy"},
        {"timestamp": 1700000010000, "price": 103.0, "quantity": 1, "side": "buy"},
        {"timestamp": 1700000020000, "price": 98.0, "quantity": 1, "side": "sell"},
        {"timestamp": 1700000030000, "price": 101.0, "quantity": 1, "side": "buy"},
    ]
    
    for event in events:
        agg.process_event(event)
    
    current = agg.get_current_candle()
    assert current["open"] == 100.0
    assert current["high"] == 103.0
    assert current["low"] == 98.0
    assert current["close"] == 101.0
    assert current["volume"] == 4.0
    assert current["delta"] == 2.0  # 3 buys (3.0) - 1 sell (1.0) = 2.0


def test_get_current_candle_immutable():
    """Test that get_current_candle returns a copy not reference."""
    agg = CandleAggregator("1m")
    agg.process_event({
        "timestamp": 1700000000000,
        "price": 100.0,
        "quantity": 1.0,
        "side": "buy"
    })
    
    candle_copy = agg.get_current_candle()
    candle_copy["close"] = 999.99  # Modify returned copy
    
    # Internal state should remain unchanged
    assert agg.get_current_candle()["close"] != 999.99


def test_reset_functionality():
    """Test that reset clears aggregator state."""
    agg = CandleAggregator("1m")
    agg.process_event({
        "timestamp": 1700000000000,
        "price": 100.0,
        "quantity": 1.0,
        "side": "buy"
    })
    
    assert agg.get_current_candle() is not None
    
    agg.reset()
    
    assert agg.get_current_candle() is None
    assert agg._current_bucket is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])