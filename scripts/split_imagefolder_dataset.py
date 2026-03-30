from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split an ImageFolder-style dataset into train/test directories.\n\n"
            "Expected input layout:\n"
            "  <source_root>/<class_name>/<image files>\n\n"
            "Output layout:\n"
            "  <output_root>/train/<class_name>/<image files>\n"
            "  <output_root>/test/<class_name>/<image files>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "source_root",
        type=Path,
        help="Source dataset root containing one subdirectory per class.",
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help="Destination root where train/ and test/ will be created.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of each class assigned to test. Default: 0.2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for deterministic splitting. Default: 7",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the output root first if it already exists.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset root not found: {source_root}")
    if not source_root.is_dir():
        raise NotADirectoryError(f"Source dataset root is not a directory: {source_root}")
    if not 0.0 < args.test_ratio < 1.0:
        raise ValueError("--test-ratio must be between 0 and 1.")

    class_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(
            f"No class directories found under source root: {source_root}"
        )

    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"Output root already exists: {output_root}. "
                "Pass --force to replace it."
            )
        shutil.rmtree(output_root)

    rng = random.Random(args.seed)
    train_root = output_root / "train"
    test_root = output_root / "test"

    total_train = 0
    total_test = 0

    for class_dir in class_dirs:
        images = sorted(
            path for path in class_dir.iterdir() if path.is_file() and is_image_file(path)
        )
        if not images:
            continue

        shuffled = list(images)
        rng.shuffle(shuffled)

        test_count = max(1, int(round(len(shuffled) * args.test_ratio)))
        if test_count >= len(shuffled):
            test_count = len(shuffled) - 1

        test_files = shuffled[:test_count]
        train_files = shuffled[test_count:]

        if not train_files or not test_files:
            raise ValueError(
                f"Class {class_dir.name} does not have enough images for a "
                f"train/test split with test_ratio={args.test_ratio}."
            )

        copy_group(
            train_files,
            source_root=source_root,
            destination_root=train_root / class_dir.name,
            do_copy=args.copy,
        )
        copy_group(
            test_files,
            source_root=source_root,
            destination_root=test_root / class_dir.name,
            do_copy=args.copy,
        )

        total_train += len(train_files)
        total_test += len(test_files)

        print(
            f"{class_dir.name}: train={len(train_files)} test={len(test_files)}"
        )

    print("")
    print(f"Created split under: {output_root}")
    print(f"Train images: {total_train}")
    print(f"Test images: {total_test}")
    print(f"Mode: {'copy' if args.copy else 'symlink'}")
    return 0


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def copy_group(
    files: list[Path],
    *,
    source_root: Path,
    destination_root: Path,
    do_copy: bool,
) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    for source_path in files:
        target_path = destination_root / source_path.name
        if do_copy:
            shutil.copy2(source_path, target_path)
        else:
            relative_target = Path(
                os.path.relpath(source_path, start=target_path.parent)
            )
            target_path.symlink_to(relative_target)


if __name__ == "__main__":
    raise SystemExit(main())
