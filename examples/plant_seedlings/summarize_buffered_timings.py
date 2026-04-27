#!/usr/bin/env python3
"""Summarize buffered-vs-sync timing runs from a key=value timings file.

Expected input lines look like:

buffered=true epochs=3 batch_size=32 num_workers=2 elapsed_seconds=246.892510
buffered=false epochs=3 batch_size=32 num_workers=2 elapsed_seconds=250.405925
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TimingRow:
    buffered: bool
    elapsed_seconds: float
    raw_line: str


def parse_line(line: str) -> TimingRow | None:
    text = line.strip()
    if not text:
        return None

    fields: dict[str, str] = {}
    for token in text.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value

    if "buffered" not in fields or "elapsed_seconds" not in fields:
        return None

    buffered_value = fields["buffered"].strip().lower()
    if buffered_value == "true":
        buffered = True
    elif buffered_value == "false":
        buffered = False
    else:
        return None

    try:
        elapsed_seconds = float(fields["elapsed_seconds"])
    except ValueError:
        return None

    return TimingRow(
        buffered=buffered,
        elapsed_seconds=elapsed_seconds,
        raw_line=text,
    )


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {}

    mean_value = statistics.mean(values)
    median_value = statistics.median(values)
    min_value = min(values)
    max_value = max(values)
    stddev_value = statistics.stdev(values) if len(values) > 1 else 0.0

    return {
        "count": float(len(values)),
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
        "median": median_value,
        "stddev": stddev_value,
    }


def format_seconds(seconds: float) -> str:
    return f"{seconds:.6f}s"


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty list")
    if q <= 0.0:
        return sorted_values[0]
    if q >= 1.0:
        return sorted_values[-1]

    index = (len(sorted_values) - 1) * q
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return sorted_values[lower]

    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def bootstrap_ci_mean(values: list[float], *, samples: int, seed: int) -> tuple[float, float] | None:
    if not values:
        return None

    rng = random.Random(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(samples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(draw))

    means.sort()
    lo = percentile(means, 0.025)
    hi = percentile(means, 0.975)
    return lo, hi


def print_group(title: str, values: list[float]) -> None:
    print(f"\n{title}")
    if not values:
        print("  no runs")
        return

    stats = describe(values)
    print(f"  count  : {int(stats['count'])}")
    print(f"  min    : {format_seconds(stats['min'])}")
    print(f"  max    : {format_seconds(stats['max'])}")
    print(f"  mean   : {format_seconds(stats['mean'])}")
    print(f"  median : {format_seconds(stats['median'])}")
    print(f"  stddev : {format_seconds(stats['stddev'])}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read timing lines and print buffered-vs-sync statistics."
    )
    parser.add_argument(
        "--timing-file",
        type=Path,
        default=Path("examples/plant_seedlings/timings_buffered_compare.txt"),
        help="Path to buffered timing file.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Bootstrap resamples used for paired CI. Default: 10000",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=7,
        help="Random seed for bootstrap CI. Default: 7",
    )
    args = parser.parse_args()

    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be >= 1")

    timing_path = args.timing_file.expanduser().resolve()
    if not timing_path.exists():
        raise FileNotFoundError(f"Timing file does not exist: {timing_path}")

    rows: list[TimingRow] = []
    skipped = 0

    for line in timing_path.read_text(encoding="utf-8").splitlines():
        row = parse_line(line)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    with_buffered = [r.elapsed_seconds for r in rows if r.buffered]
    without_buffered = [r.elapsed_seconds for r in rows if not r.buffered]

    print(f"timing_file={timing_path}")
    print(f"parsed_runs={len(rows)} skipped_lines={skipped}")

    print_group("Buffered = True", with_buffered)
    print_group("Buffered = False (sync)", without_buffered)

    if with_buffered and without_buffered:
        mean_with = statistics.mean(with_buffered)
        mean_without = statistics.mean(without_buffered)
        delta = mean_with - mean_without
        pct = (delta / mean_without) * 100.0 if not math.isclose(mean_without, 0.0) else math.nan

        print("\nComparison (mean)")
        print(f"  buffered - sync : {delta:+.6f}s")
        if math.isnan(pct):
            print("  percent change  : n/a")
        else:
            print(f"  percent change  : {pct:+.2f}%")

    pair_count = min(len(with_buffered), len(without_buffered))
    if pair_count > 0:
        paired_with = with_buffered[:pair_count]
        paired_without = without_buffered[:pair_count]
        paired_deltas = [w - wo for w, wo in zip(paired_with, paired_without)]
        paired_pct = [
            ((w - wo) / wo) * 100.0
            for w, wo in zip(paired_with, paired_without)
            if not math.isclose(wo, 0.0)
        ]

        print("\nPaired Deltas (buffered_i - sync_i)")
        print(f"  pairs  : {pair_count}")
        print(f"  mean   : {format_seconds(statistics.mean(paired_deltas))}")
        print(f"  median : {format_seconds(statistics.median(paired_deltas))}")
        if len(paired_deltas) > 1:
            print(f"  stddev : {format_seconds(statistics.stdev(paired_deltas))}")
        else:
            print("  stddev : 0.000000s")
        print(f"  min    : {format_seconds(min(paired_deltas))}")
        print(f"  max    : {format_seconds(max(paired_deltas))}")

        if paired_pct:
            print(f"  mean % : {statistics.mean(paired_pct):+.2f}%")

        ci_delta = bootstrap_ci_mean(
            paired_deltas,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        if ci_delta is not None:
            print(
                "  95% CI (mean delta): "
                f"[{format_seconds(ci_delta[0])}, {format_seconds(ci_delta[1])}]"
            )

        if paired_pct:
            ci_pct = bootstrap_ci_mean(
                paired_pct,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            )
            if ci_pct is not None:
                print(
                    "  95% CI (mean %): "
                    f"[{ci_pct[0]:+.2f}%, {ci_pct[1]:+.2f}%]"
                )

        print("\nPair Details")
        for idx, (w, wo) in enumerate(zip(paired_with, paired_without), start=1):
            delta_seconds = w - wo
            if math.isclose(wo, 0.0):
                pct_text = "n/a"
            else:
                pct_text = f"{((delta_seconds / wo) * 100.0):+.2f}%"
            print(
                f"  pair {idx:2d}: "
                f"buffered={format_seconds(w)} "
                f"sync={format_seconds(wo)} "
                f"delta={delta_seconds:+.6f}s "
                f"({pct_text})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
