#!/usr/bin/env python3
"""Summarize instrumentation timing runs from a key=value timings file.

Expected input lines look like:

instrumentation=True epochs=3 batch_size=32 num_workers=2 elapsed_seconds=246.892510
instrumentation=False epochs=3 batch_size=32 num_workers=2 elapsed_seconds=250.405925
"""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TimingRow:
    instrumentation: bool
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

    if "instrumentation" not in fields or "elapsed_seconds" not in fields:
        return None

    instrumentation_value = fields["instrumentation"].strip().lower()
    if instrumentation_value == "true":
        instrumentation = True
    elif instrumentation_value == "false":
        instrumentation = False
    else:
        return None

    try:
        elapsed_seconds = float(fields["elapsed_seconds"])
    except ValueError:
        return None

    return TimingRow(
        instrumentation=instrumentation,
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
        description="Read timing lines and print instrumentation statistics."
    )
    parser.add_argument(
        "--timing-file",
        type=Path,
        default=Path("examples/plant_seedlings/timings.txt"),
        help="Path to timings file. Default: examples/plant_seedlings/timings.txt",
    )
    args = parser.parse_args()

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

    with_inst = [r.elapsed_seconds for r in rows if r.instrumentation]
    without_inst = [r.elapsed_seconds for r in rows if not r.instrumentation]

    print(f"timing_file={timing_path}")
    print(f"parsed_runs={len(rows)} skipped_lines={skipped}")

    print_group("Instrumentation = True", with_inst)
    print_group("Instrumentation = False", without_inst)

    if with_inst and without_inst:
        mean_with = statistics.mean(with_inst)
        mean_without = statistics.mean(without_inst)
        delta = mean_with - mean_without
        pct = (delta / mean_without) * 100.0 if not math.isclose(mean_without, 0.0) else math.nan

        print("\nComparison (mean)")
        print(f"  with - without : {delta:+.6f}s")
        if math.isnan(pct):
            print("  percent change : n/a")
        else:
            print(f"  percent change : {pct:+.2f}%")

    print("\nRaw runs")
    if not rows:
        print("  no valid timing rows found")
    else:
        for row in rows:
            label = "with" if row.instrumentation else "without"
            print(f"  {label:7} {format_seconds(row.elapsed_seconds)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
