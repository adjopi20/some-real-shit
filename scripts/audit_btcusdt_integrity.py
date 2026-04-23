import glob
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import pandas as pd
import pyarrow.parquet as pq


def fmt_wib(ms: int) -> str:
    wib = timezone(timedelta(hours=7))
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone(wib).strftime(
        "%Y-%m-%d %H:%M:%S WIB"
    )


def main() -> None:
    base = os.path.join("storage", "btcusdt")
    files = sorted(glob.glob(os.path.join(base, "*.parquet")))
    if not files:
        print("No parquet files found in storage/btcusdt")
        return

    conn = sqlite3.connect("_tmp_btcusdt_audit.db")
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=OFF;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("CREATE TABLE IF NOT EXISTS seen_local (id INTEGER PRIMARY KEY)")
    cur.execute("CREATE TABLE IF NOT EXISTS seen_trade (id INTEGER PRIMARY KEY)")
    conn.commit()

    rows_total = 0
    min_ts = 10**30
    max_ts = -1

    schema_counter = Counter()
    dtype_variants = defaultdict(set)
    null_counts_total = Counter()

    side_invalid_total = 0
    price_non_positive_total = 0
    qty_non_positive_total = 0

    files_internal_ts_disorder = []
    files_internal_local_disorder = []
    files_internal_trade_disorder = []

    within_dup_local_total = 0
    within_dup_trade_total = 0
    cross_dup_local_total = 0
    cross_dup_trade_total = 0

    file_ranges = []

    for i, f in enumerate(files, start=1):
        table = pq.read_table(f)
        df = table.to_pandas()

        rows = len(df)
        rows_total += rows

        schema_counter[tuple(df.columns)] += 1
        for c in df.columns:
            dtype_variants[c].add(str(df[c].dtype))

        nulls = df.isna().sum()
        for c, n in nulls.items():
            if int(n):
                null_counts_total[c] += int(n)

        if "timestamp" in df.columns:
            fmin = int(df["timestamp"].min())
            fmax = int(df["timestamp"].max())
            min_ts = min(min_ts, fmin)
            max_ts = max(max_ts, fmax)
            file_ranges.append((os.path.basename(f), fmin, fmax, rows))
            if not df["timestamp"].is_monotonic_increasing:
                files_internal_ts_disorder.append(os.path.basename(f))

        if "local_id" in df.columns:
            within_dup_local_total += int(df["local_id"].duplicated().sum())
            local_vals = pd.to_numeric(df["local_id"], errors="coerce").dropna().astype("int64")
            if not local_vals.is_monotonic_increasing:
                files_internal_local_disorder.append(os.path.basename(f))
            before = conn.total_changes
            cur.executemany(
                "INSERT OR IGNORE INTO seen_local(id) VALUES (?)",
                ((int(v),) for v in local_vals.tolist()),
            )
            conn.commit()
            inserted = conn.total_changes - before
            cross_dup_local_total += len(local_vals) - inserted

        if "trade_id" in df.columns:
            within_dup_trade_total += int(df["trade_id"].duplicated().sum())
            trade_vals = pd.to_numeric(df["trade_id"], errors="coerce").dropna().astype("int64")
            if not trade_vals.is_monotonic_increasing:
                files_internal_trade_disorder.append(os.path.basename(f))
            before = conn.total_changes
            cur.executemany(
                "INSERT OR IGNORE INTO seen_trade(id) VALUES (?)",
                ((int(v),) for v in trade_vals.tolist()),
            )
            conn.commit()
            inserted = conn.total_changes - before
            cross_dup_trade_total += len(trade_vals) - inserted

        if "side" in df.columns:
            side_invalid_total += int((~df["side"].isin([-1, 1])).sum())
        if "price" in df.columns:
            price_non_positive_total += int((pd.to_numeric(df["price"], errors="coerce") <= 0).sum())
        if "quantity" in df.columns:
            qty_non_positive_total += int((pd.to_numeric(df["quantity"], errors="coerce") <= 0).sum())

        if i % 500 == 0:
            print(f"Processed {i}/{len(files)} files...")

    file_ranges_sorted = sorted(file_ranges, key=lambda x: x[1])
    segments = []
    current = [file_ranges_sorted[0]]
    for r in file_ranges_sorted[1:]:
        prev = current[-1]
        if r[1] - prev[2] > 5 * 60 * 1000:
            segments.append(current)
            current = [r]
        else:
            current.append(r)
    segments.append(current)

    boundary_backwards = 0
    boundary_exact_overlap = 0
    boundary_gap_gt1s = 0
    max_backward_ms = 0
    max_forward_ms = 0

    for a, b in zip(file_ranges_sorted, file_ranges_sorted[1:]):
        delta = b[1] - a[2]
        if delta < 0:
            boundary_backwards += 1
            max_backward_ms = min(max_backward_ms, delta)
        elif delta == 0:
            boundary_exact_overlap += 1
        elif delta > 1000:
            boundary_gap_gt1s += 1
            max_forward_ms = max(max_forward_ms, delta)

    print("\n===== BTCUSDT INTEGRITY AUDIT =====")
    print(f"Total files: {len(files)}")
    print(f"Total rows: {rows_total}")
    print(f"Time range: {fmt_wib(min_ts)} -> {fmt_wib(max_ts)}")

    print("\nSchema variants:")
    for cols, cnt in schema_counter.items():
        print(f"  - {cnt} files: {cols}")

    print("\nDtype variants:")
    for c, v in sorted(dtype_variants.items()):
        print(f"  - {c}: {sorted(v)}")

    print("\nData quality checks:")
    print(f"  - Null counts: {dict(null_counts_total)}")
    print(f"  - Invalid side values (!= -1/1): {side_invalid_total}")
    print(f"  - Non-positive price rows: {price_non_positive_total}")
    print(f"  - Non-positive quantity rows: {qty_non_positive_total}")

    print("\nDuplicate checks:")
    print(f"  - Within-file duplicated local_id rows: {within_dup_local_total}")
    print(f"  - Cross-file/global duplicated local_id rows: {cross_dup_local_total}")
    print(f"  - Within-file duplicated trade_id rows: {within_dup_trade_total}")
    print(f"  - Cross-file/global duplicated trade_id rows: {cross_dup_trade_total}")

    print("\nOrdering checks:")
    print(f"  - Files with internal timestamp disorder: {len(files_internal_ts_disorder)}")
    print(f"  - Files with internal local_id disorder: {len(files_internal_local_disorder)}")
    print(f"  - Files with internal trade_id disorder: {len(files_internal_trade_disorder)}")
    print(f"  - File boundary backward overlaps: {boundary_backwards}")
    print(f"  - File boundary exact overlaps (next.min_ts == prev.max_ts): {boundary_exact_overlap}")
    print(f"  - File boundary forward gaps >1s: {boundary_gap_gt1s}")
    print(f"  - Max backward overlap (ms): {max_backward_ms}")
    print(f"  - Max forward gap >1s (ms): {max_forward_ms}")

    print("\nTimeline segments (gap >5min starts new segment):")
    for idx, seg in enumerate(segments, start=1):
        print(
            f"  - Segment {idx}: {len(seg)} files, {fmt_wib(seg[0][1])} -> {fmt_wib(seg[-1][2])}, first={seg[0][0]}, last={seg[-1][0]}"
        )

    conn.close()
    if os.path.exists("_tmp_btcusdt_audit.db"):
        os.remove("_tmp_btcusdt_audit.db")


if __name__ == "__main__":
    main()