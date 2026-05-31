#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "results" / "overhead" / "figures"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.plant_seedlings.summarize_timings import parse_line


@dataclass(frozen=True)
class TimingSummary:
    label: str
    with_values: list[float]
    without_values: list[float]

    @property
    def mean_with(self) -> float:
        return statistics.mean(self.with_values)

    @property
    def mean_without(self) -> float:
        return statistics.mean(self.without_values)

    @property
    def mean_delta(self) -> float:
        return self.mean_with - self.mean_without

    @property
    def paired_pct(self) -> list[float]:
        return [
            ((with_value - without_value) / without_value) * 100.0
            for with_value, without_value in zip(self.with_values, self.without_values)
        ]

    @property
    def mean_pct(self) -> float:
        return statistics.mean(self.paired_pct)


def parse_sweep_log(path: Path) -> list[TimingSummary]:
    summaries: list[TimingSummary] = []
    current_batch_size: str | None = None
    collecting_timings = False
    with_values: list[float] = []
    without_values: list[float] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if line.startswith("Running covtype batch-size sweep for batch_size="):
            current_batch_size = line.rsplit("=", 1)[-1]
            with_values = []
            without_values = []
            collecting_timings = False
            continue

        if line == "Done. Timings:":
            collecting_timings = True
            continue

        if collecting_timings and line == "Summary:":
            if current_batch_size is None:
                raise ValueError(f"Found timing block without batch size in {path}")
            if not with_values or not without_values:
                raise ValueError(
                    f"Incomplete timing block for batch_size={current_batch_size} in {path}"
                )
            summaries.append(
                TimingSummary(
                    label=current_batch_size,
                    with_values=with_values,
                    without_values=without_values,
                )
            )
            collecting_timings = False
            continue

        if collecting_timings:
            row = parse_line(line)
            if row is None:
                continue
            if row.instrumentation:
                with_values.append(row.elapsed_seconds)
            else:
                without_values.append(row.elapsed_seconds)

    if collecting_timings and current_batch_size is not None:
        summaries.append(
            TimingSummary(
                label=current_batch_size,
                with_values=with_values,
                without_values=without_values,
            )
        )

    if not summaries:
        raise ValueError(f"No sweep timing blocks parsed from {path}")

    return summaries


def read_timings(path: Path, *, label: str) -> TimingSummary:
    with_values: list[float] = []
    without_values: list[float] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        row = parse_line(line)
        if row is None:
            continue
        if row.instrumentation:
            with_values.append(row.elapsed_seconds)
        else:
            without_values.append(row.elapsed_seconds)

    if not with_values or not without_values:
        raise ValueError(f"No timing rows parsed from {path}")
    if len(with_values) != len(without_values):
        raise ValueError(
            f"Mismatched paired timings in {path}: "
            f"{len(with_values)} instrumented vs {len(without_values)} baseline"
        )

    return TimingSummary(label=label, with_values=with_values, without_values=without_values)


def ci95_halfwidth(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * (statistics.stdev(values) / math.sqrt(len(values)))


def make_main_overhead_figure(
    plant: TimingSummary,
    covtype: TimingSummary,
    output_dir: Path,
) -> None:
    labels = [plant.label, covtype.label]
    baseline_means = [plant.mean_without, covtype.mean_without]
    instrumented_means = [plant.mean_with, covtype.mean_with]
    overhead_pct = [plant.mean_pct, covtype.mean_pct]

    x = range(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.bar(
        [i - width / 2 for i in x],
        baseline_means,
        width,
        label="Baseline",
        color="#8ea3b0",
    )
    ax.bar(
        [i + width / 2 for i in x],
        instrumented_means,
        width,
        label="PyPyrus",
        color="#2b6f8a",
    )

    ax.set_ylabel("Mean runtime (s)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title("Runtime overhead by workload")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    y_max = max(instrumented_means)
    ax.set_ylim(0, y_max * 1.18)

    for idx, pct in enumerate(overhead_pct):
        y_top = max(baseline_means[idx], instrumented_means[idx])
        ax.text(
            idx,
            y_top + y_max * 0.045,
            f"+{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_dir / "main_overhead_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "main_overhead_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def make_batchsize_sweep_figure(
    summaries: list[TimingSummary],
    output_dir: Path,
) -> None:
    batch_sizes = [int(summary.label) for summary in summaries]
    mean_pct = [summary.mean_pct for summary in summaries]
    mean_delta = [summary.mean_delta for summary in summaries]

    fig, ax1 = plt.subplots(figsize=(8.6, 5.2))
    ax1.plot(
        batch_sizes,
        mean_pct,
        color="#8c4f66",
        marker="o",
        linewidth=2,
        label="Relative overhead (%)",
    )
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Mean overhead (%)", color="#8c4f66")
    ax1.tick_params(axis="y", labelcolor="#8c4f66")
    ax1.set_xticks(batch_sizes)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.plot(
        batch_sizes,
        mean_delta,
        color="#436436",
        marker="s",
        linewidth=2,
        label="Absolute overhead (s)",
    )
    ax2.set_ylabel("Mean overhead (s)", color="#436436")
    ax2.tick_params(axis="y", labelcolor="#436436")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=False, loc="upper right")
    ax1.set_title("Covtype overhead sensitivity to batch size")

    fig.tight_layout()
    fig.savefig(output_dir / "covtype_batchsize_sweep.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "covtype_batchsize_sweep.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary_table(
    plant: TimingSummary,
    covtype: TimingSummary,
    sweep_summaries: list[TimingSummary],
    output_dir: Path,
) -> None:
    lines = [
        "figure\tlabel\tmean_baseline_s\tmean_pypyrus_s\tmean_delta_s\tmean_overhead_pct",
        (
            "main\tplant\t"
            f"{plant.mean_without:.6f}\t{plant.mean_with:.6f}\t"
            f"{plant.mean_delta:.6f}\t{plant.mean_pct:.2f}"
        ),
        (
            "main\tcovtype\t"
            f"{covtype.mean_without:.6f}\t{covtype.mean_with:.6f}\t"
            f"{covtype.mean_delta:.6f}\t{covtype.mean_pct:.2f}"
        ),
    ]
    for summary in sweep_summaries:
        lines.append(
            "batchsize\t"
            f"{summary.label}\t{summary.mean_without:.6f}\t{summary.mean_with:.6f}\t"
            f"{summary.mean_delta:.6f}\t{summary.mean_pct:.2f}"
        )
    (output_dir / "figure_data.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate thesis-ready overhead figures from current overhead timing artifacts."
    )
    parser.add_argument(
        "--plant-timings",
        type=Path,
        default=REPO_ROOT / "experiments" / "results" / "overhead" / "plant_timings.txt",
        help="Path to the current plant overhead timings file.",
    )
    parser.add_argument(
        "--covtype-timings",
        type=Path,
        default=REPO_ROOT / "experiments" / "results" / "overhead" / "covtype_timings.txt",
        help="Path to the current covtype overhead timings file.",
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "results" / "overhead" / "batchsize_sweep",
        help="Directory containing covtype batch-size sweep timing files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures and derived summary data will be written.",
    )
    parser.add_argument(
        "--sweep-log",
        type=Path,
        default=None,
        help="Optional batch-size sweep log to parse directly, e.g. logs/pypyrus_overhead_bsweep_28492468.out",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plant = read_timings(args.plant_timings.expanduser().resolve(), label="Plant seedlings")
    covtype = read_timings(args.covtype_timings.expanduser().resolve(), label="Forest Covertype")

    if args.sweep_log is not None:
        sweep_summaries = parse_sweep_log(args.sweep_log.expanduser().resolve())
        sweep_summaries.sort(key=lambda summary: int(summary.label))
    else:
        sweep_summaries = []
        for batch_size in (64, 128, 256, 512, 1024):
            timing_path = args.sweep_dir.expanduser().resolve() / f"covtype_bs{batch_size}_timings.txt"
            sweep_summaries.append(read_timings(timing_path, label=str(batch_size)))

    make_main_overhead_figure(plant, covtype, output_dir)
    make_batchsize_sweep_figure(sweep_summaries, output_dir)
    write_summary_table(plant, covtype, sweep_summaries, output_dir)

    print(f"Wrote figures to: {output_dir}")
    print(f"  - {output_dir / 'main_overhead_comparison.png'}")
    print(f"  - {output_dir / 'main_overhead_comparison.pdf'}")
    print(f"  - {output_dir / 'covtype_batchsize_sweep.png'}")
    print(f"  - {output_dir / 'covtype_batchsize_sweep.pdf'}")
    print(f"  - {output_dir / 'figure_data.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
