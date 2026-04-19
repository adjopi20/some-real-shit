"""Lightweight schema definitions for raw trade collection."""

from __future__ import annotations

from typing import Any

# Internal in-memory tuple layout (minimal, numeric-heavy):
# (timestamp_ms, recv_ts_ns, local_id, price, quantity, side)
TRADE_TIMESTAMP_IDX = 0
TRADE_RECV_TS_IDX = 1
TRADE_LOCAL_ID_IDX = 2
TRADE_PRICE_IDX = 3
TRADE_QUANTITY_IDX = 4
TRADE_SIDE_IDX = 5

SCHEMA_VERSION = "v1"


def parquet_schema() -> Any:
    """Output parquet schema (exact required data contract)."""
    try:
        import pyarrow as pa
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyarrow is required for parquet schema generation. "
            "Install it with: pip install pyarrow"
        ) from exc

    return pa.schema(
        [
            ("schema_version", pa.string()),
            ("timestamp", pa.int64()),
            ("recv_ts", pa.int64()),
            ("local_id", pa.int64()),
            ("price", pa.float64()),
            ("quantity", pa.float64()),
            ("side", pa.int8()),
        ]
    )
