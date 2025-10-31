#!/usr/bin/env python3
"""Deterministic cross-validation fold assignment for GDPa1 tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path
from typing import Iterable, List


def _stable_fold(identifier: str, seed: str, n_folds: int) -> int:
    """Hash an identifier+seed pair into a fold index."""

    digest = hashlib.sha256(f"{seed}::{identifier}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n_folds


def _resolve_output(input_path: Path, output_path: Path, overwrite: bool) -> Path:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")
    if input_path == output_path:
        return output_path.with_suffix(output_path.suffix + ".tmp")
    return output_path


def _write_with_folds(
    input_path: Path,
    output_path: Path,
    id_column: str,
    hash_columns: List[str],
    fold_column: str,
    seed: str,
    n_folds: int,
    validate_column: str | None,
) -> int:
    mismatches = 0
    with input_path.open("r", newline="", encoding="utf-8") as src, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        if not reader.fieldnames or id_column not in reader.fieldnames:
            raise KeyError(f"Identifier column '{id_column}' not found in {input_path}")

        fieldnames = list(reader.fieldnames)
        if fold_column not in fieldnames:
            fieldnames.append(fold_column)

        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            identifier = row.get(id_column, "").strip()
            if not identifier:
                raise ValueError(f"Row missing identifier '{id_column}': {row}")

            key_parts = [identifier]
            for column in hash_columns:
                if column not in row:
                    raise KeyError(f"Hash column '{column}' not found for identifier {identifier}")
                key_parts.append(row[column].strip())

            composite_key = "::".join(key_parts)
            fold_value = str(_stable_fold(composite_key, seed, n_folds))
            row[fold_column] = fold_value

            if validate_column is not None:
                existing = row.get(validate_column)
                if existing is None:
                    raise KeyError(
                        f"Validation column '{validate_column}' missing for identifier {identifier}"
                    )
                if existing.strip() != fold_value:
                    mismatches += 1

            writer.writerow(row)

    return mismatches


def assign_folds(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    candidate_output = _resolve_output(input_path, output_path, args.overwrite)
    mismatch_count = _write_with_folds(
        input_path=input_path,
        output_path=candidate_output,
        id_column=args.id_column,
        hash_columns=args.hash_columns or [],
        fold_column=args.fold_column,
        seed=args.seed,
        n_folds=args.num_folds,
        validate_column=args.validate_column,
    )

    if candidate_output != output_path:
        shutil.move(str(candidate_output), str(output_path))

    if args.validate_column and mismatch_count:
        message = (
            f"Deterministic folds disagree with column '{args.validate_column}' for {mismatch_count} rows"
        )
        if args.strict:
            raise ValueError(message)
        print(message, file=sys.stderr)

    if args.verbose:
        print(
            {
                "input": str(input_path),
                "output": str(output_path),
                "fold_column": args.fold_column,
                "num_folds": args.num_folds,
                "seed": args.seed,
                "mismatches": mismatch_count,
            }
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assign deterministic cross-validation folds to GDPa1 tables.",
    )
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Destination CSV path")
    parser.add_argument(
        "--id-column",
        default="antibody_id",
        help="Column used as the deterministic key (default: antibody_id)",
    )
    parser.add_argument(
        "--fold-column",
        default="deterministic_fold",
        help="Column to create or overwrite with deterministic folds",
    )
    parser.add_argument(
        "--hash-columns",
        action="append",
        help="Additional columns to mix into the hash key (appendable).",
    )
    parser.add_argument(
        "--validate-column",
        default=None,
        help="Optional existing column to check for equality with the deterministic fold",
    )
    parser.add_argument(
        "--seed",
        default="bitcore-gdpa1",
        help="Seed string mixed into the hash so folds stay reproducible",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Number of folds to generate (default: 5)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error when validation mismatches are detected.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a summary of the assignment run",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        assign_folds(args)
    except Exception as exc:  # noqa: BLE001
        parser.error(str(exc))


if __name__ == "__main__":
    main()