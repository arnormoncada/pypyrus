#!/usr/bin/env python3
"""Small baseline timing script for PyPyrus storage modes.

This benchmark compares four scenarios:
1. raw: no PyPyrus instrumentation
2. pypyrus_noop_sync: instrumentation with a no-op store (event prep baseline)
3. pypyrus_sqlite_sync: instrumentation with synchronous SQLite writes
4. pypyrus_sqlite_buffered: instrumentation with buffered strict writer

Use this to estimate where overhead comes from before deeper optimization.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from pypyrus import Run, attach
from pypyrus.provenance.events import ProvenanceEvent
from pypyrus.storage import SQLiteStore, Store


class SyntheticDataset(Dataset):
    def __init__(self, n: int, feature_dim: int) -> None:
        self._x = torch.randn(n, feature_dim)
        self._y = torch.randint(low=0, high=10, size=(n,))

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self._x[idx], int(self._y[idx].item())


class NoopStore(Store):
    """Store implementation used to estimate non-DB instrumentation overhead."""

    def __init__(self) -> None:
        self._events = 0

    def initialize(self) -> None:
        return None

    def close(self) -> None:
        return None

    def append_event(self, event: ProvenanceEvent) -> None:
        self._events += 1

    def flush(self) -> None:
        return None

    def get_events(self, run_id: str, event_type: str | None = None) -> list[dict[str, Any]]:
        return []

    def list_runs(self) -> list[str]:
        return []


def iterate_all_batches(loader: DataLoader) -> int:
    seen = 0
    for batch in loader:
        # Touch the batch payload so the loop does real work.
        _ = batch
        seen += 1
    return seen


def make_loader(
    *,
    samples: int,
    feature_dim: int,
    batch_size: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = SyntheticDataset(samples, feature_dim)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )


def run_raw_trial(*, samples: int, feature_dim: int, batch_size: int, seed: int) -> tuple[float, int]:
    loader = make_loader(
        samples=samples,
        feature_dim=feature_dim,
        batch_size=batch_size,
        seed=seed,
    )

    t0 = time.perf_counter()
    batches = iterate_all_batches(loader)
    elapsed = time.perf_counter() - t0
    return elapsed, batches


def run_noop_trial(
    *,
    samples: int,
    feature_dim: int,
    batch_size: int,
    seed: int,
    store_mode: str,
    buffered_queue_size: int,
) -> tuple[float, int]:
    loader = make_loader(
        samples=samples,
        feature_dim=feature_dim,
        batch_size=batch_size,
        seed=seed,
    )

    store = NoopStore()
    with Run(
        store=store,
        store_mode=store_mode,
        buffered_queue_size=buffered_queue_size,
    ) as run:
        attached = attach(loader, run, role="train")
        t0 = time.perf_counter()
        batches = iterate_all_batches(attached)
        elapsed = time.perf_counter() - t0

    return elapsed, batches


def run_sqlite_trial(
    *,
    samples: int,
    feature_dim: int,
    batch_size: int,
    seed: int,
    store_mode: str,
    buffered_queue_size: int,
    db_path: Path,
) -> tuple[float, int]:
    loader = make_loader(
        samples=samples,
        feature_dim=feature_dim,
        batch_size=batch_size,
        seed=seed,
    )

    store = SQLiteStore(db_path)
    with Run(
        store=store,
        store_mode=store_mode,
        buffered_queue_size=buffered_queue_size,
    ) as run:
        attached = attach(loader, run, role="train")
        t0 = time.perf_counter()
        batches = iterate_all_batches(attached)
        elapsed = time.perf_counter() - t0

    return elapsed, batches


def summarize(name: str, values: list[float], baseline_mean: float | None) -> None:
    mean = statistics.mean(values)
    median = statistics.median(values)
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0

    line = (
        f"{name:24s} mean={mean:.6f}s median={median:.6f}s "
        f"std={stddev:.6f}s n={len(values)}"
    )
    if baseline_mean is not None and baseline_mean > 0:
        delta = mean - baseline_mean
        pct = (delta / baseline_mean) * 100.0
        line += f" delta_vs_raw={delta:+.6f}s ({pct:+.2f}%)"
    print(line)


def run_once(
    *,
    samples: int,
    feature_dim: int,
    batch_size: int,
    seed: int,
    buffered_queue_size: int,
    temp_dir: Path,
) -> dict[str, tuple[float, int]]:
    sqlite_sync_db = temp_dir / f"sync_{seed}.db"
    sqlite_buffered_db = temp_dir / f"buffered_{seed}.db"

    return {
        "raw": run_raw_trial(
            samples=samples,
            feature_dim=feature_dim,
            batch_size=batch_size,
            seed=seed,
        ),
        "pypyrus_noop_sync": run_noop_trial(
            samples=samples,
            feature_dim=feature_dim,
            batch_size=batch_size,
            seed=seed,
            store_mode="sync",
            buffered_queue_size=buffered_queue_size,
        ),
        "pypyrus_noop_buffered": run_noop_trial(
            samples=samples,
            feature_dim=feature_dim,
            batch_size=batch_size,
            seed=seed,
            store_mode="buffered_strict",
            buffered_queue_size=buffered_queue_size,
        ),
        "pypyrus_sqlite_sync": run_sqlite_trial(
            samples=samples,
            feature_dim=feature_dim,
            batch_size=batch_size,
            seed=seed,
            store_mode="sync",
            buffered_queue_size=buffered_queue_size,
            db_path=sqlite_sync_db,
        ),
        "pypyrus_sqlite_buffered": run_sqlite_trial(
            samples=samples,
            feature_dim=feature_dim,
            batch_size=batch_size,
            seed=seed,
            store_mode="buffered_strict",
            buffered_queue_size=buffered_queue_size,
            db_path=sqlite_buffered_db,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline timing for PyPyrus sync vs buffered store modes")
    parser.add_argument("--samples", type=int, default=20000, help="Number of samples in synthetic dataset")
    parser.add_argument("--feature-dim", type=int, default=32, help="Feature dimension for synthetic tensors")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--repeats", type=int, default=5, help="Number of timing trials")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed")
    parser.add_argument("--buffered-queue-size", type=int, default=1024, help="Queue size for buffered_strict mode")
    parser.add_argument(
        "--keep-dbs",
        action="store_true",
        help="Keep generated sqlite benchmark DB files for inspection",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    with tempfile.TemporaryDirectory(prefix="pypyrus_baseline_timing_") as tmp:
        temp_dir = Path(tmp)
        by_mode: dict[str, list[float]] = {
            "raw": [],
            "pypyrus_noop_sync": [],
            "pypyrus_noop_buffered": [],
            "pypyrus_sqlite_sync": [],
            "pypyrus_sqlite_buffered": [],
        }

        for trial in range(args.repeats):
            trial_seed = args.seed + trial
            result = run_once(
                samples=args.samples,
                feature_dim=args.feature_dim,
                batch_size=args.batch_size,
                seed=trial_seed,
                buffered_queue_size=args.buffered_queue_size,
                temp_dir=temp_dir,
            )

            # Sanity check: all modes should iterate same number of batches.
            batch_counts = {name: batches for name, (_, batches) in result.items()}
            if len(set(batch_counts.values())) != 1:
                raise RuntimeError(f"Batch count mismatch across modes: {batch_counts}")

            for name, (elapsed, _) in result.items():
                by_mode[name].append(elapsed)

            print(f"trial={trial + 1}/{args.repeats} seed={trial_seed} done")

        print("\n=== Baseline Timing Summary ===")
        print(
            f"samples={args.samples} batch_size={args.batch_size} "
            f"feature_dim={args.feature_dim} repeats={args.repeats}"
        )
        raw_mean = statistics.mean(by_mode["raw"])

        summarize("raw", by_mode["raw"], baseline_mean=None)
        summarize("pypyrus_noop_sync", by_mode["pypyrus_noop_sync"], baseline_mean=raw_mean)
        summarize("pypyrus_noop_buffered", by_mode["pypyrus_noop_buffered"], baseline_mean=raw_mean)
        summarize("pypyrus_sqlite_sync", by_mode["pypyrus_sqlite_sync"], baseline_mean=raw_mean)
        summarize("pypyrus_sqlite_buffered", by_mode["pypyrus_sqlite_buffered"], baseline_mean=raw_mean)

        print("\nInterpretation hint:")
        print("- noop_* approximates event-preparation overhead without DB write cost")
        print("- sqlite_* adds durable persistence cost")
        print("- sqlite_sync vs sqlite_buffered estimates write-path offloading impact")

        if args.keep_dbs:
            keep_dir = Path.cwd() / "tmp_baseline_timing_dbs"
            keep_dir.mkdir(parents=True, exist_ok=True)
            for child in temp_dir.iterdir():
                target = keep_dir / child.name
                if target.exists():
                    if target.is_file():
                        target.unlink()
                os.replace(child, target)
            print(f"\nSaved benchmark DB files to: {keep_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
