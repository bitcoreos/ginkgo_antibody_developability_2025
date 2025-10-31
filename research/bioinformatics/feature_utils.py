"""Shared amino-acid property utilities for feature engineering modules."""

from collections import Counter
import math
import re
from typing import Dict, Iterable, List

HYDROPHOBICITY_SCALE: Dict[str, float] = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

POSITIVE_CHARGE_WEIGHTS: Dict[str, float] = {'K': 1.0, 'R': 1.0, 'H': 0.5}
NEGATIVE_CHARGE_WEIGHTS: Dict[str, float] = {'D': -1.0, 'E': -1.0}
CHARGE_SCALE: Dict[str, float] = {
    'D': -1.0,
    'E': -1.0,
    'K': 1.0,
    'R': 1.0,
    'H': 0.5
}
FLEXIBLE_RESIDUES = set('GSTDNQ')
AROMATIC_RESIDUES = set('FWYH')
GLYCOSYLATION_PATTERN = re.compile(r'N[^P][ST]')


def hydrophobicity_values(sequence: str, scale: Dict[str, float] = HYDROPHOBICITY_SCALE) -> List[float]:
    """Return hydrophobicity values for a sequence using the provided scale."""
    if not sequence:
        return []
    return [scale.get(aa, 0.0) for aa in sequence]


def shannon_entropy(sequence: str) -> float:
    """Compute Shannon entropy of amino-acid composition."""
    if not sequence:
        return 0.0
    counts = Counter(sequence)
    length = len(sequence)
    entropy = 0.0
    for count in counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)
    return entropy


def charge_totals(sequence: str) -> Dict[str, float]:
    """Return positive, negative, and net charge totals for a sequence."""
    positive = sum(POSITIVE_CHARGE_WEIGHTS.get(aa, 0.0) for aa in sequence)
    negative = -sum(NEGATIVE_CHARGE_WEIGHTS.get(aa, 0.0) for aa in sequence)
    return {
        'positive': positive,
        'negative': negative,
        'net': positive - negative
    }


def count_glycosylation_sites(sequence: str) -> int:
    """Count N-linked glycosylation motifs (N-X-S/T, X != P)."""
    if not sequence:
        return 0
    return len(GLYCOSYLATION_PATTERN.findall(sequence))


def residue_fraction(sequence: str, residues: Iterable[str]) -> float:
    """Return the fraction of residues belonging to the provided set."""
    if not sequence:
        return 0.0
    residue_set = set(residues)
    return sum(1 for aa in sequence if aa in residue_set) / len(sequence)
