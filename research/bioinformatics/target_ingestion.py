#!/usr/bin/env python3
"""GDPa1 target assay ingestion.

Parses the raw GDPa1 target table, aligns antibody identifiers with the
sequence reference sheet, and emits a clean target matrix ready for modeling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class TargetIngestion:
    """Load, validate, and persist GDPa1 target assays."""

    workspace_root: Path = Path("/workspaces/bitcore/workspace")
    raw_filename: str = "GDPa1_v1.2_20250814.csv"
    output_filename: str = "gdpa1_competition_targets.csv"

    RAW_TO_STANDARD: Dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.data_dir = self.workspace_root / "data"
        self.raw_dir = self.data_dir / "raw"
        self.sequences_file = self.data_dir / "sequences" / "GDPa1_v1.2_sequences.csv"
        self.output_dir = self.data_dir / "targets"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_file = self.raw_dir / self.raw_filename
        self.output_file = self.output_dir / self.output_filename

        if self.RAW_TO_STANDARD is None:
            self.RAW_TO_STANDARD = {
                "Titer": "Titer_g/L",
                "HIC": "HIC_delta_G_ML",
                "AC-SINS_pH7.4": "AC-SINS_pH7.4_nmol/mg",
                "Tm2": "Tm2_DSF_degC",
                "PR_CHO": "PR_CHO",
            }

        self.standard_columns: List[str] = [
            "antibody_id",
            "antibody_name",
            *self.RAW_TO_STANDARD.values(),
        ]

    def load_raw_targets(self) -> pd.DataFrame:
        """Read the raw CSV and keep only the target assay fields."""
        if not self.raw_file.exists():
            raise FileNotFoundError(f"Raw target file not found: {self.raw_file}")

        df = pd.read_csv(self.raw_file)
        required_columns = {"antibody_id", "antibody_name", *self.RAW_TO_STANDARD.keys()}
        missing = required_columns.difference(df.columns)
        if missing:
            raise ValueError(f"Missing expected columns in raw file: {sorted(missing)}")

        df = df[list(required_columns)].copy()
        df = df.rename(columns=self.RAW_TO_STANDARD)

        for col in self.RAW_TO_STANDARD.values():
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["antibody_id"] = df["antibody_id"].astype(str)
        df = df.sort_values("antibody_id").reset_index(drop=True)
        return df

    def validate_against_sequences(self, targets_df: pd.DataFrame) -> None:
        """Ensure that target rows match the reference sequence listing."""
        if not self.sequences_file.exists():
            raise FileNotFoundError(f"Sequence reference not found: {self.sequences_file}")

        seq_df = pd.read_csv(self.sequences_file, usecols=["antibody_id"])
        sequences = set(seq_df["antibody_id"].astype(str))
        targets = set(targets_df["antibody_id"])

        missing_in_targets = sorted(sequences.difference(targets))
        if missing_in_targets:
            logger.warning("%d antibodies missing target assays", len(missing_in_targets))

        extra_in_targets = sorted(targets.difference(sequences))
        if extra_in_targets:
            raise ValueError(f"Unexpected antibody ids in target file: {extra_in_targets}")

    def save_targets(self, targets_df: pd.DataFrame) -> Path:
        """Persist the cleaned target matrix to disk."""
        targets_df.to_csv(self.output_file, index=False)
        logger.info("Saved cleaned targets to %s", self.output_file)
        return self.output_file

    def run(self) -> pd.DataFrame:
        """Execute the full ingestion workflow."""
        targets_df = self.load_raw_targets()
        self.validate_against_sequences(targets_df)
        self.save_targets(targets_df)
        return targets_df


def main() -> None:
    ingestion = TargetIngestion()
    targets_df = ingestion.run()

    available = targets_df.drop(columns=["antibody_name"]).set_index("antibody_id").notna().sum()
    logger.info("Target coverage summary: %s", available.to_dict())


if __name__ == "__main__":  # pragma: no cover
    main()
