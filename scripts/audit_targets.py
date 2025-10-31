#!/usr/bin/env python3
"""Audit GDPa1 target assays for completeness and provenance logging."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
TARGETS_PATH = WORKSPACE_ROOT / "data" / "targets" / "gdpa1_competition_targets.csv"
SEQUENCES_PATH = WORKSPACE_ROOT / "data" / "sequences" / "GDPa1_v1.2_sequences.csv"
AUDIT_DIR = WORKSPACE_ROOT / "data" / "targets" / "audit"

REQUIRED_COLUMNS: List[str] = [
    "AC-SINS_pH7.4_nmol/mg",
    "Titer_g/L",
    "antibody_id",
    "HIC_delta_G_ML",
    "PR_CHO",
    "Tm2_DSF_degC",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4096), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Expected file missing: {path}")
    return pd.read_csv(path)


def audit_targets(target_path: Path, sequences_path: Path) -> Dict[str, object]:
    targets = _load_csv(target_path)
    sequences = _load_csv(sequences_path)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in targets.columns]
    duplicate_ids = targets["antibody_id"].duplicated().sum()

    seq_ids = set(sequences["antibody_id"].astype(str))
    target_ids = set(targets["antibody_id"].astype(str))

    missing_in_targets = sorted(seq_ids - target_ids)
    extra_in_targets = sorted(target_ids - seq_ids)

    coverage = {
        column: int(targets[column].notna().sum()) if column in targets.columns else 0
        for column in REQUIRED_COLUMNS
    }

    fold_alignment = None
    if "hierarchical_cluster_IgG_isotype_stratified_fold" in sequences.columns:
        merged = targets.merge(
            sequences[["antibody_id", "hierarchical_cluster_IgG_isotype_stratified_fold"]],
            on="antibody_id",
            how="left",
        )
        unique_folds = sorted(
            int(value)
            for value in merged["hierarchical_cluster_IgG_isotype_stratified_fold"].dropna().unique()
        )
        fold_alignment = {
            "null_folds": int(
                merged["hierarchical_cluster_IgG_isotype_stratified_fold"].isna().sum()
            ),
            "unique_folds": unique_folds,
        }

    stats: Dict[str, object] = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
        "targets_path": str(target_path),
        "targets_sha256": _sha256(target_path),
        "targets_rows": int(len(targets)),
        "targets_columns": list(targets.columns),
        "required_columns_missing": missing_columns,
        "duplicate_antibody_ids": int(duplicate_ids),
        "coverage_counts": coverage,
        "missing_in_targets": missing_in_targets,
        "extra_in_targets": extra_in_targets,
        "fold_alignment": fold_alignment,
    }

    return stats


def _write_audit_log(stats: Dict[str, object], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"gdpa1_targets_audit_{timestamp}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit GDPa1 competition target assays")
    parser.add_argument(
        "--targets",
        default=str(TARGETS_PATH),
        help="Path to gdpa1_competition_targets.csv",
    )
    parser.add_argument(
        "--sequences",
        default=str(SEQUENCES_PATH),
        help="Path to GDPa1_v1.2_sequences.csv",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Skip writing JSON audit log and only print summary",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    target_path = Path(args.targets).resolve()
    sequences_path = Path(args.sequences).resolve()

    stats = audit_targets(target_path, sequences_path)

    print(json.dumps(stats, indent=2))

    if not args.no_log:
        log_path = _write_audit_log(stats, AUDIT_DIR)
        print(f"Audit log saved to {log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
