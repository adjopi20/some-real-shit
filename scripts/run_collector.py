import asyncio
import signal
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collector import config
from collector.binance_ws import BinanceTradeStream
from collector.writer import BufferedParquetWriter


def _log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")


async def _stats_loop(writer: BufferedParquetWriter, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        await asyncio.sleep(config.STATS_LOG_INTERVAL_SEC)
        _log(
            "stats "
            f"buffer={len(writer.active_buffer)} "
            f"written_events={writer.total_events_written} "
            f"chunks={writer.total_chunks_written}"
        )


async def main() -> None:
    stop_event = asyncio.Event()

    stream = BinanceTradeStream()
    writer = BufferedParquetWriter(logger=_log)

    async def request_shutdown() -> None:
        if not stop_event.is_set():
            _log("shutdown requested")
            stop_event.set()

    loop = asyncio.get_running_loop()

    # Cross-platform signal handling.
    def _signal_callback() -> None:
        asyncio.create_task(request_shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_callback)
        except NotImplementedError:
            signal.signal(sig, lambda *_: asyncio.create_task(request_shutdown()))

    periodic_flush_task = asyncio.create_task(writer.periodic_flush_loop(stop_event))
    stats_task = asyncio.create_task(_stats_loop(writer, stop_event))
    stream_task = asyncio.create_task(stream.stream(writer.add_event, logger=_log, stop_event=stop_event))

    done, pending = await asyncio.wait(
        {stream_task, periodic_flush_task, stats_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    # If a background task exits unexpectedly, trigger graceful shutdown.
    if not stop_event.is_set():
        for task in done:
            if task.exception() is not None:
                _log(f"task exited with error: {task.exception()}")
        stop_event.set()

    for task in pending:
        task.cancel()

    await asyncio.gather(*pending, return_exceptions=True)
    await writer.shutdown()
    _log("collector stopped cleanly")


if __name__ == "__main__":
    asyncio.run(main())
