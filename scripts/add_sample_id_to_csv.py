from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUT_PATH = Path("experiments/forest_covertype/data/covtype.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Add a sample_id column to a CSV with values obs_1, obs_2, "
            "obs_3, ..."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=(
            "Path to the input CSV. "
            f"Default: {DEFAULT_INPUT_PATH}"
        ),
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        type=Path,
        help=(
            "Path to the output CSV. "
            "Defaults to <input_stem>_with_sample_id.csv next to the input file."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    input_path = args.input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path = resolve_output_path(input_path, args.output_path)
    add_sample_id_column(input_path, output_path)

    print(f"Wrote CSV with sample_id column to: {output_path}")
    return 0


def resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path.expanduser().resolve()
    return input_path.with_name(f"{input_path.stem}_with_sample_id{input_path.suffix}")


def add_sample_id_column(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["sample_id", *reader.fieldnames]

        with output_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()

            for row_number, row in enumerate(reader, start=1):
                writer.writerow(
                    {
                        "sample_id": f"obs_{row_number}",
                        **row,
                    }
                )


if __name__ == "__main__":
    raise SystemExit(main())
