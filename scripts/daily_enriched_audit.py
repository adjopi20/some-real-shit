import argparse
import bisect
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REQUIRED_FLOW_FIELDS = [
    "timestamp",
    "price",
    "delta",
    "window_cum_delta",
    "session_cum_delta",
    "buy_volume",
    "sell_volume",
    "pressure",
    "volume_per_sec",
    "delta_per_sec",
    "trade_intensity",
    "price_range",
    "absorption",
    "warm",
]

NUMERIC_FLOW_FIELDS = [
    "timestamp",
    "price",
    "delta",
    "window_cum_delta",
    "session_cum_delta",
    "buy_volume",
    "sell_volume",
    "pressure",
    "volume_per_sec",
    "delta_per_sec",
    "trade_intensity",
    "price_range",
]


def iso_ms(ts_ms: float) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()


def quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = max(0, min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def pearson_corr(x: List[float], y: List[float]) -> float:
    n = min(len(x), len(y))
    if n < 3:
        return float("nan")
    x = x[:n]
    y = y[:n]
    mx = statistics.fmean(x)
    my = statistics.fmean(y)
    sx = math.sqrt(sum((v - mx) ** 2 for v in x))
    sy = math.sqrt(sum((v - my) ** 2 for v in y))
    if sx == 0 or sy == 0:
        return float("nan")
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / (sx * sy)


def permutation_pvalue_diff_means(
    a: List[float], b: List[float], n_iter: int = 2000, seed: int = 42
) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    rng = random.Random(seed)
    obs = abs(statistics.fmean(a) - statistics.fmean(b))
    merged = a + b
    na = len(a)
    ge = 0
    for _ in range(n_iter):
        rng.shuffle(merged)
        d = abs(statistics.fmean(merged[:na]) - statistics.fmean(merged[na:]))
        if d >= obs:
            ge += 1
    return (ge + 1) / (n_iter + 1)


@dataclass
class RuleViolation:
    count: int = 0
    lines: List[int] = field(default_factory=list)

    def add(self, line_no: int):
        self.count += 1
        if len(self.lines) < 10:
            self.lines.append(line_no)


def audit_file(input_path: Path) -> Dict:
    violations: Dict[str, RuleViolation] = defaultdict(RuleViolation)
    keysets = Counter()
    field_presence = Counter()
    type_map = defaultdict(Counter)

    ts_list: List[int] = []
    price_list: List[float] = []
    pressure_list: List[float] = []
    intensity_list: List[float] = []
    range_list: List[float] = []
    absorption_list: List[bool] = []

    future_ready_prices: List[Tuple[int, float, float, bool, float]] = []

    parsed = 0
    json_errors = 0
    empty_lines = 0
    prev_ts: Optional[int] = None
    ts_counter = Counter()

    with input_path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            s = raw.strip()
            if not s:
                empty_lines += 1
                violations["empty_line"].add(i)
                continue

            try:
                obj = json.loads(s)
            except Exception:
                json_errors += 1
                violations["json_parse_error"].add(i)
                continue

            parsed += 1

            if not isinstance(obj, dict):
                violations["top_level_not_object"].add(i)
                continue

            keysets[tuple(sorted(obj.keys()))] += 1

            if "ts" not in obj:
                violations["missing_top_ts"].add(i)
                continue
            if "flow" not in obj:
                violations["missing_top_flow"].add(i)
                continue

            ts = obj["ts"]
            flow = obj["flow"]

            if not isinstance(ts, int):
                violations["ts_not_int"].add(i)
                continue
            if not isinstance(flow, dict):
                violations["flow_not_object"].add(i)
                continue

            if prev_ts is not None and ts < prev_ts:
                violations["ts_non_monotonic"].add(i)
            prev_ts = ts

            ts_counter[ts] += 1
            ts_list.append(ts)

            for k, v in flow.items():
                field_presence[k] += 1
                type_map[k][type(v).__name__] += 1

            for req in REQUIRED_FLOW_FIELDS:
                if req not in flow:
                    violations[f"missing_flow_{req}"].add(i)

            if any(req not in flow for req in REQUIRED_FLOW_FIELDS):
                continue

            for nf in NUMERIC_FLOW_FIELDS:
                if not isinstance(flow[nf], (int, float)) or isinstance(flow[nf], bool):
                    violations[f"non_numeric_{nf}"].add(i)
                    continue
                if not math.isfinite(float(flow[nf])):
                    violations[f"non_finite_{nf}"].add(i)

            if not isinstance(flow["absorption"], bool):
                violations["absorption_not_bool"].add(i)
            if not isinstance(flow["warm"], bool):
                violations["warm_not_bool"].add(i)

            if float(flow["timestamp"]) != float(ts):
                violations["flow_timestamp_mismatch_top_ts"].add(i)

            price = float(flow["price"])
            buy_v = float(flow["buy_volume"])
            sell_v = float(flow["sell_volume"])
            pressure = float(flow["pressure"])
            window_delta = float(flow["window_cum_delta"])
            trade_intensity = float(flow["trade_intensity"])
            price_range = float(flow["price_range"])

            if price <= 0:
                violations["price_non_positive"].add(i)
            if buy_v < 0 or sell_v < 0:
                violations["negative_volume"].add(i)
            if price_range < 0:
                violations["negative_price_range"].add(i)
            if pressure < -1.000001 or pressure > 1.000001:
                violations["pressure_out_of_bounds"].add(i)

            total = buy_v + sell_v
            if total <= 0:
                violations["non_positive_total_volume"].add(i)
            else:
                expected_pressure = window_delta / total
                if abs(expected_pressure - pressure) > 2e-6:
                    violations["pressure_formula_mismatch"].add(i)

            if not flow["warm"]:
                violations["warm_false_in_saved_file"].add(i)

            price_list.append(price)
            pressure_list.append(pressure)
            intensity_list.append(trade_intensity)
            range_list.append(price_range)
            absorption_list.append(bool(flow["absorption"]))
            future_ready_prices.append((ts, price, pressure, bool(flow["absorption"]), price_range))

    # Availability and timing
    gaps = []
    for a, b in zip(ts_list[:-1], ts_list[1:]):
        gaps.append(b - a)

    duplicate_rows = sum(v - 1 for v in ts_counter.values() if v > 1)
    max_same_ts = max(ts_counter.values()) if ts_counter else 0

    # Hypothesis feature engineering with time horizons
    ts_only = [t for t, *_ in future_ready_prices]
    prices_only = [p for _, p, *_ in future_ready_prices]
    pressures_only = [pr for _, _, pr, *_ in future_ready_prices]
    absorptions_only = [ab for _, _, _, ab, _ in future_ready_prices]
    ranges_only = [rg for *_, rg in future_ready_prices]

    fut_ret_3s = []
    fut_abs_ret_3s = []
    pressure_for_3s = []
    abs_pressure_for_3s = []

    fut_abs_ret_10s = []
    abs_flag_10s = []

    for i, ts in enumerate(ts_only):
        idx_3 = bisect.bisect_left(ts_only, ts + 3000, lo=i + 1)
        if idx_3 < len(ts_only):
            r3 = (prices_only[idx_3] - prices_only[i]) / prices_only[i]
            fut_ret_3s.append(r3)
            fut_abs_ret_3s.append(abs(r3))
            pressure_for_3s.append(pressures_only[i])
            abs_pressure_for_3s.append(abs(pressures_only[i]))

        idx_10 = bisect.bisect_left(ts_only, ts + 10000, lo=i + 1)
        if idx_10 < len(ts_only):
            r10 = abs((prices_only[idx_10] - prices_only[i]) / prices_only[i])
            fut_abs_ret_10s.append(r10)
            abs_flag_10s.append(absorptions_only[i])

    # H1 groups (absorption true/false)
    h1_true = [r for r, a in zip(fut_abs_ret_10s, abs_flag_10s) if a]
    h1_false = [r for r, a in zip(fut_abs_ret_10s, abs_flag_10s) if not a]

    # H2 directional accuracy under strong pressure
    threshold = 0.6
    strong_idx = [i for i, p in enumerate(pressure_for_3s) if abs(p) >= threshold]
    directional_total = len(strong_idx)
    directional_hits = 0
    for idx in strong_idx:
        pred = 1 if pressure_for_3s[idx] > 0 else -1
        realized = 1 if fut_ret_3s[idx] > 0 else (-1 if fut_ret_3s[idx] < 0 else 0)
        if realized != 0 and pred == realized:
            directional_hits += 1

    # Outlier check via IQR on selected metrics
    def iqr_outliers(vals: List[float]) -> Tuple[float, float, int]:
        if len(vals) < 8:
            return float("nan"), float("nan"), 0
        sv = sorted(vals)
        q1 = quantile(sv, 0.25)
        q3 = quantile(sv, 0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        count = sum(1 for v in vals if v < lo or v > hi)
        return lo, hi, count

    outlier_summary = {
        "delta": iqr_outliers([float(v) for v in [json.loads(line)["flow"]["delta"] for line in input_path.open("r", encoding="utf-8") if line.strip()]]),
        "pressure": iqr_outliers(pressure_list),
        "trade_intensity": iqr_outliers(intensity_list),
        "price_range": iqr_outliers(range_list),
    }

    report = {
        "input_file": str(input_path),
        "rows_total": parsed + json_errors + empty_lines,
        "rows_parsed": parsed,
        "rows_empty": empty_lines,
        "rows_json_error": json_errors,
        "unique_keysets": len(keysets),
        "top_keysets": keysets.most_common(5),
        "field_presence": field_presence,
        "type_map": type_map,
        "violations": violations,
        "ts_min": min(ts_list) if ts_list else None,
        "ts_max": max(ts_list) if ts_list else None,
        "gaps": gaps,
        "duplicate_rows": duplicate_rows,
        "max_same_ts": max_same_ts,
        "price_list": price_list,
        "pressure_list": pressure_list,
        "intensity_list": intensity_list,
        "range_list": range_list,
        "absorption_list": absorption_list,
        "h1_true": h1_true,
        "h1_false": h1_false,
        "fut_ret_3s": fut_ret_3s,
        "fut_abs_ret_3s": fut_abs_ret_3s,
        "pressure_for_3s": pressure_for_3s,
        "abs_pressure_for_3s": abs_pressure_for_3s,
        "directional_total": directional_total,
        "directional_hits": directional_hits,
        "ranges_only": ranges_only,
        "outlier_summary": outlier_summary,
    }
    return report


def fmt_stats(vals: List[float]) -> str:
    if not vals:
        return "count=0"
    sv = sorted(vals)
    return (
        f"count={len(vals)} | min={sv[0]:.8f} | p01={quantile(sv,0.01):.8f} | "
        f"p50={statistics.median(sv):.8f} | p99={quantile(sv,0.99):.8f} | max={sv[-1]:.8f}"
    )


def build_report_text(r: Dict) -> str:
    ts_min = r["ts_min"]
    ts_max = r["ts_max"]
    gaps = r["gaps"]

    gap_stats = "count=0"
    gap_over_1s = gap_over_5s = gap_over_10s = gap_over_60s = 0
    if gaps:
        s = sorted(gaps)
        gap_stats = (
            f"count={len(gaps)} | min={s[0]}ms | p50={statistics.median(s):.2f}ms | "
            f"p95={quantile(s,0.95):.2f}ms | max={s[-1]}ms"
        )
        gap_over_1s = sum(1 for g in gaps if g > 1000)
        gap_over_5s = sum(1 for g in gaps if g > 5000)
        gap_over_10s = sum(1 for g in gaps if g > 10000)
        gap_over_60s = sum(1 for g in gaps if g > 60000)

    duration_s = ((ts_max - ts_min) / 1000.0) if (ts_min is not None and ts_max is not None) else 0
    rows_per_sec = (r["rows_parsed"] / duration_s) if duration_s > 0 else float("nan")

    v_items = sorted(r["violations"].items(), key=lambda kv: (-kv[1].count, kv[0]))
    violation_lines = []
    for name, info in v_items:
        if info.count > 0:
            violation_lines.append(f"- {name}: count={info.count}, sample_lines={info.lines}")
    if not violation_lines:
        violation_lines = ["- none"]

    h1_true = r["h1_true"]
    h1_false = r["h1_false"]
    h1_mean_t = statistics.fmean(h1_true) if h1_true else float("nan")
    h1_mean_f = statistics.fmean(h1_false) if h1_false else float("nan")
    h1_ratio = (h1_mean_t / h1_mean_f) if h1_false and h1_mean_f != 0 else float("nan")
    h1_p = permutation_pvalue_diff_means(h1_true, h1_false, n_iter=2000)

    c_pressure_ret3 = pearson_corr(r["pressure_for_3s"], r["fut_ret_3s"])
    c_abs_pressure_absret3 = pearson_corr(r["abs_pressure_for_3s"], r["fut_abs_ret_3s"])
    c_intensity_range = pearson_corr(r["intensity_list"], r["range_list"])

    dir_acc = (
        r["directional_hits"] / r["directional_total"]
        if r["directional_total"] > 0
        else float("nan")
    )

    out = []
    out.append("DAILY ENRICHED DATA AUDIT REPORT (STANDARD FORMAT)")
    out.append("=" * 72)
    out.append(f"Input file: {r['input_file']}")
    out.append(f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}")
    out.append("")

    out.append("1) DATA INTEGRITY (LINE-BY-LINE RULE AUDIT)")
    out.append("-" * 72)
    out.append(f"Rows total           : {r['rows_total']}")
    out.append(f"Rows parsed          : {r['rows_parsed']}")
    out.append(f"Rows empty           : {r['rows_empty']}")
    out.append(f"Rows JSON errors     : {r['rows_json_error']}")
    out.append(f"Unique top-level schemas: {r['unique_keysets']}")
    out.append(f"Duplicate timestamp rows : {r['duplicate_rows']} (max rows sharing same ts: {r['max_same_ts']})")
    out.append("Rule violations summary:")
    out.extend(violation_lines)
    out.append("")

    out.append("2) DATA AVAILABILITY")
    out.append("-" * 72)
    if ts_min is not None and ts_max is not None:
        out.append(f"Time coverage (UTC)  : {iso_ms(ts_min)} -> {iso_ms(ts_max)}")
        out.append(f"Coverage duration    : {duration_s:.2f} seconds")
        out.append(f"Rows per second      : {rows_per_sec:.2f}")
    else:
        out.append("Time coverage        : unavailable")
    out.append(f"Inter-row gap stats  : {gap_stats}")
    out.append(f"Gap > 1s             : {gap_over_1s}")
    out.append(f"Gap > 5s             : {gap_over_5s}")
    out.append(f"Gap > 10s            : {gap_over_10s}")
    out.append(f"Gap > 60s            : {gap_over_60s}")
    out.append("")

    out.append("3) DATA RELIABILITY / DISTRIBUTION HEALTH")
    out.append("-" * 72)
    out.append(f"price                : {fmt_stats(r['price_list'])}")
    out.append(f"pressure             : {fmt_stats(r['pressure_list'])}")
    out.append(f"trade_intensity      : {fmt_stats(r['intensity_list'])}")
    out.append(f"price_range          : {fmt_stats(r['range_list'])}")
    out.append("IQR outlier counts (lo, hi, count):")
    for k, (lo, hi, count) in r["outlier_summary"].items():
        out.append(f"- {k}: lo={lo:.8f}, hi={hi:.8f}, count={count}")
    out.append("")

    out.append("4) HYPOTHESES & TEST RESULTS")
    out.append("-" * 72)
    out.append("H1: Absorption flags are associated with larger future volatility (10s horizon).")
    out.append(
        f"- mean(|ret_10s|) absorption=True: {h1_mean_t:.8f} (n={len(h1_true)}) | "
        f"absorption=False: {h1_mean_f:.8f} (n={len(h1_false)})"
    )
    out.append(f"- ratio True/False: {h1_ratio:.4f} | permutation p-value: {h1_p:.6f}")

    out.append("H2: Pressure has directional relationship with short-horizon returns (3s).")
    out.append(f"- corr(pressure_t, ret_3s): {c_pressure_ret3:.6f}")
    out.append(
        f"- directional accuracy when |pressure|>=0.6: {dir_acc:.4f} "
        f"(hits={r['directional_hits']}, total={r['directional_total']})"
    )

    out.append("H3: Trade intensity and price range co-move (activity vs micro-volatility).")
    out.append(f"- corr(trade_intensity, price_range): {c_intensity_range:.6f}")

    out.append("H4: Stronger imbalance magnitude precedes larger absolute returns (3s).")
    out.append(f"- corr(|pressure_t|, |ret_3s|): {c_abs_pressure_absret3:.6f}")
    out.append("")

    out.append("5) INTERPRETATION GUIDELINES (FOR DAILY COMPARISON)")
    out.append("-" * 72)
    out.append("- Integrity PASS baseline: zero JSON errors, zero schema/type violations.")
    out.append("- Availability warning baseline: any gap > 10s should be flagged operationally.")
    out.append("- Hypothesis tracking:")
    out.append("  * H1 stronger when ratio True/False > 1 and p-value < 0.05.")
    out.append("  * H2 directional utility improves as accuracy rises meaningfully above 0.50.")
    out.append("  * H3/H4 monitor correlation drift over time for regime changes.")

    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Daily audit for enriched JSONL flow files.")
    parser.add_argument("--input", required=True, help="Path to input enriched JSONL file")
    parser.add_argument("--output", required=True, help="Path to output TXT report")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_data = audit_file(input_path)
    report_text = build_report_text(report_data)
    output_path.write_text(report_text, encoding="utf-8")
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
