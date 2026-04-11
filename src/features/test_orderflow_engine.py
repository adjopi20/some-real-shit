import pytest
from orderflow_engine import OrderFlowEngine


def make_event(ts, price, qty, side):
    return {
        "timestamp": ts,
        "price": price,
        "quantity": qty,
        "side": side
    }


# =========================
# 1. DELTA CONSISTENCY
# =========================
def test_delta_accumulation_and_rolloff():
    engine = OrderFlowEngine(window_type="count", window_size=3)

    engine.process_event(make_event(1, 100, 1, "buy"))   # +1
    engine.process_event(make_event(2, 100, 2, "sell"))  # -2
    engine.process_event(make_event(3, 100, 3, "buy"))   # +3

    assert engine._window_cum_delta == 1 - 2 + 3

    # This event will push out the first (+1)
    engine.process_event(make_event(4, 100, 1, "buy"))
    engine.process_event(make_event(5, 100, 2, "buy"))

    # Expected: (-2 + 3 + 1)
    assert engine._window_cum_delta == (-2 + 3 + 1 + 2)


# =========================
# 2. WINDOW RESET (CRITICAL)
# =========================
def test_stale_reset_clears_state():
    engine = OrderFlowEngine()

    engine.process_event(make_event(0, 100, 1, "buy"))
    engine.process_event(make_event(1000, 100, 1, "buy"))

    # Large gap → triggers reset
    engine.process_event(make_event(20000, 100, 1, "buy"))

    assert len(engine._events) == 1
    assert engine._window_cum_delta == 1
    assert engine._last_event_ts == 20000


# =========================
# 3. WARM FLAG
# =========================
def test_warm_flag_behavior():
    engine = OrderFlowEngine(window_type="count", window_size=10)

    outputs = []
    for i in range(5):
        out = engine.process_event(make_event(i, 100, 1, "buy"))
        outputs.append(out)

    # Early events should not be warm until baseline updates
    assert any(not o["warm"] for o in outputs)

    # Push more events to stabilize baseline
    for i in range(5, 15):
        out = engine.process_event(make_event(i, 100, 1, "buy"))

    assert out["warm"] is True


# =========================
# 4. ABSORPTION DETECTION
# =========================
def test_absorption_triggers_on_high_volume_low_range():
    engine = OrderFlowEngine(window_type="count", window_size=10)

    # Build baseline (low volume)
    for i in range(10):
        engine.process_event(make_event(i, 100, 1, "buy"))

    # Now inject high volume but tight price
    for i in range(10, 11):
        out = engine.process_event(make_event(i, 100.0001, 10, "buy"))

    assert out["absorption"] is True


# =========================
# 5. BASELINE FRESHNESS
# =========================
def test_baseline_staleness_blocks_absorption():
    engine = OrderFlowEngine(window_type="time", window_size=1)

    # Build baseline
    for i in range(5):
        engine.process_event(make_event(i * 1000, 100, 1, "buy"))

    # Wait too long → baseline stale
    out = engine.process_event(make_event(100000, 100, 50, "buy"))

    assert out["absorption"] is False


# =========================
# 6. PRESSURE RANGE
# =========================
def test_pressure_bounds():
    engine = OrderFlowEngine()

    engine.process_event(make_event(1, 100, 10, "buy"))
    engine.process_event(make_event(2, 100, 10, "sell"))

    out = engine.process_event(make_event(3, 100, 5, "buy"))

    assert -1.0 <= out["pressure"] <= 1.0


# =========================
# 7. NO DIVISION ERRORS
# =========================
def test_no_division_by_zero():
    engine = OrderFlowEngine()

    out = engine.process_event(make_event(1, 100, 1, "buy"))

    assert out["volume_per_sec"] == 0.0
    assert out["delta_per_sec"] == 0.0