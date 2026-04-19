"""High-performance buffered parquet writer for raw trade events."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from collector import config, schema

TradeTuple = tuple[int, int, int, float, float, int]
LogCallback = Optional[Callable[[str], None]]


class BufferedParquetWriter:
    """Write-optimized append-only parquet chunk writer.

    Behavior:
      ingest -> append -> swap -> async write -> continue ingest
    """

    def __init__(
        self,
        output_dir: str = config.OUTPUT_DIR,
        chunk_size: int = config.CHUNK_SIZE,
        flush_interval_sec: int = config.FLUSH_INTERVAL_SEC,
        logger: LogCallback = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_size = chunk_size
        self.flush_interval_sec = flush_interval_sec
        self.logger = logger

        self.active_buffer: list[TradeTuple] = []
        self.pending_tasks: set[asyncio.Task] = set()
        self._is_flushing = False
        self._shutting_down = False

        self._chunk_index = 0
        self._invalid_count = 0
        self.total_events_buffered = 0
        self.total_events_written = 0
        self.total_chunks_written = 0

    def add_event(self, event: TradeTuple) -> None:
        """Fast non-blocking ingestion path (no locks, no I/O)."""
        timestamp_ms, _, _, price, quantity, _ = event

        # Data integrity safeguard before buffering.
        if timestamp_ms <= 0 or price <= 0.0 or quantity <= 0.0:
            self._invalid_count += 1
            return

        self.active_buffer.append(event)
        self.total_events_buffered += 1

        # Soft backpressure warning only (never drop events).
        if self.logger and len(self.active_buffer) >= (2 * self.chunk_size):
            self.logger(f"buffer warning: size={len(self.active_buffer)}")

        # Never block event loop; schedule flush in background.
        if len(self.active_buffer) >= self.chunk_size and not self._is_flushing:
            asyncio.create_task(self.flush())

    async def flush(self) -> None:
        """Swap active buffer and write swapped buffer asynchronously."""
        if self._is_flushing:
            return

        if not self.active_buffer:
            return

        self._is_flushing = True

        # O(1) immediate swap.
        buffer_to_flush = self.active_buffer
        self.active_buffer = []

        if not buffer_to_flush:
            self._is_flushing = False
            return

        self._chunk_index += 1
        chunk_index = self._chunk_index

        task = asyncio.create_task(asyncio.to_thread(self._write_parquet, buffer_to_flush, chunk_index))
        self.pending_tasks.add(task)
        task.add_done_callback(self._on_flush_done)

    def _on_flush_done(self, task: asyncio.Task) -> None:
        self.pending_tasks.discard(task)

        try:
            written_rows = task.result()
            if written_rows > 0:
                self.total_events_written += written_rows
                self.total_chunks_written += 1
        except Exception as exc:
            if self.logger:
                self.logger(f"write error: {exc}")
        finally:
            self._is_flushing = False

        # If backlog built while flushing, continue draining asynchronously.
        if self.active_buffer and not self._shutting_down:
            asyncio.create_task(self.flush())

    def _write_parquet(self, events: list[TradeTuple], chunk_index: int) -> int:
        """Synchronous parquet write executed in background thread."""
        if not events:
            return 0

        schema_version_col = pa.array([schema.SCHEMA_VERSION] * len(events), type=pa.string())
        ts_col = pa.array([e[0] for e in events], type=pa.int64())
        recv_col = pa.array([e[1] for e in events], type=pa.int64())
        local_id_col = pa.array([e[2] for e in events], type=pa.int64())
        price_col = pa.array([e[3] for e in events], type=pa.float64())
        qty_col = pa.array([e[4] for e in events], type=pa.float64())
        side_col = pa.array([e[5] for e in events], type=pa.int8())

        table = pa.Table.from_arrays(
            [schema_version_col, ts_col, recv_col, local_id_col, price_col, qty_col, side_col],
            schema=schema.parquet_schema(),
        )

        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_name = f"chunk_{stamp}_{chunk_index:04d}"
        tmp_path = self.output_dir / f"{base_name}.tmp"
        final_path = self.output_dir / f"{base_name}.parquet"

        pq.write_table(table, tmp_path, compression="snappy")
        tmp_path.replace(final_path)
        return len(events)

    async def periodic_flush_loop(self, stop_event: asyncio.Event) -> None:
        """Periodic flush timer for low-activity periods."""
        try:
            while not stop_event.is_set():
                await asyncio.sleep(self.flush_interval_sec)
                await self.flush()
        except asyncio.CancelledError:
            raise

    async def shutdown(self) -> None:
        """Flush remaining buffered data and wait for all writes to complete."""
        self._shutting_down = True

        if self.active_buffer:
            await self.flush()

        if self.pending_tasks:
            await asyncio.gather(*self.pending_tasks)

        # If events arrived between gather scheduling boundaries, flush again.
        if self.active_buffer:
            await self.flush()
            if self.pending_tasks:
                await asyncio.gather(*self.pending_tasks)
