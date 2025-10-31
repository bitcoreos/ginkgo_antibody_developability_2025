
"""
Module for antibody isotype modeling based on sequence features.

This module implements a scoring system that uses only isotype-specific motifs
to ensure proper differentiation between isotypes.
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Define the five main antibody isotypes
ISOTYPES = ["IgG", "IgA", "IgM", "IgE", "IgD"]

# Priority order for resolving score ties (higher value preferred)
ISOTYPE_PRIORITY = {
    "IgE": 5,
    "IgM": 4,
    "IgA": 3,
    "IgD": 2,
    "IgG": 1,
}

# Only use isotype-specific motifs (excluding the common motif)
# The common motif KSCDKTHTCPPCP is present in all isotypes and causes scoring conflicts
CONSTANT_REGION_MOTIFS = {
    "IgG": [
        ("DKTHTCPPCPAPELLGG", 1.0),  # IgG-specific motif
    ],
    "IgA": [
        ("PAPNTPPTP", 1.0),  # IgA hinge repeat motif
    ],
    "IgM": [
        ("LLGLCLLCGLLAVFVIGS", 1.0),  # IgM-specific
    ],
    "IgE": [
        ("LLGLCLLCGLLAVFVIGSIKRRSG", 1.0),  # IgE-specific
    ],
    "IgD": [
        ("GQPREPQVYTLPP", 1.0),  # IgD-specific
    ]
}

# Glycosylation site patterns (N-X-S/T where X ≠ P)
glycosylation_pattern = re.compile(r"N[^P][ST]")

# Cysteine pattern
cysteine_pattern = re.compile(r"C")

@dataclass
class IsotypePrediction:
    """Dataclass to hold isotype prediction results."""
    predicted_isotype: str
    confidence: float
    features: Dict[str, List[str]]
    scores: Dict[str, float]
    raw_scores: Dict[str, float]  # Added for test compatibility


def extract_sequence_features(sequence: str) -> Dict[str, List[str]]:
    """
    Extract key features from an antibody sequence.

    Args:
        sequence: Amino acid sequence string

    Returns:
        Dictionary of extracted features
    """
    if not sequence or not sequence.isalpha():
        raise ValueError("Invalid sequence: must be non-empty and contain only letters")

    features = {
        "motifs": [],
        "glycosylation_sites": [],
        "cysteine_pairs": [],
        "c_terminal": []
    }

    # Find constant region motifs for each isotype (only specific ones)
    for isotype in ISOTYPES:
        for motif, _ in CONSTANT_REGION_MOTIFS[isotype]:
            if motif in sequence:
                features["motifs"].append(f"{isotype}_{motif}")

    # Find potential N-linked glycosylation sites
    for match in glycosylation_pattern.finditer(sequence):
        pos = match.start()
        site = f"N{pos+1}{sequence[pos+1]}{sequence[pos+2]}"
        features["glycosylation_sites"].append(site)

    # Find cysteine residues for disulfide bond prediction
    cysteine_positions = [i for i, aa in enumerate(sequence) if aa == 'C']
    # Simple pairing: pair adjacent cysteines (this is a simplification)
    for i in range(0, len(cysteine_positions) - 1, 2):
        c1 = cysteine_positions[i]
        c2 = cysteine_positions[i + 1]
        features["cysteine_pairs"].append(f"C{c1+1}-C{c2+1}")

    # Extract C-terminal sequence
    if len(sequence) >= 5:
        features["c_terminal"].append(sequence[-5:])

    return features

def calculate_isotype_score(sequence: str, isotype: str) -> float:
    """
    Calculate a score for how likely a sequence is to belong to a given isotype.

    Args:
        sequence: Amino acid sequence string
        isotype: The isotype to score (IgG, IgA, IgM, IgE, IgD)

    Returns:
        Score between 0 and 1 indicating likelihood of belonging to the isotype
    """
    if isotype not in ISOTYPES:
        raise ValueError(f"Invalid isotype: {isotype}. Must be one of {ISOTYPES}")

    # Extract features from the sequence
    features = extract_sequence_features(sequence)

    # Get the motifs for this isotype
    motifs = CONSTANT_REGION_MOTIFS[isotype]

    # Calculate score based on presence of motifs
    score = 0.0
    max_possible_score = sum(weight for _, weight in motifs)

    # Add weighted score for each motif present
    for motif, weight in motifs:
        if any(motif in feature for feature in features["motifs"]):
            score += weight

    # Normalize score to 0-1 range
    if max_possible_score > 0:
        score = score / max_possible_score

    return score

def predict_isotype(sequence: str) -> IsotypePrediction:
    """
    Predict the most likely isotype for an antibody sequence.

    Args:
        sequence: Amino acid sequence string

    Returns:
        IsotypePrediction object with results
    """
    # Extract features
    features = extract_sequence_features(sequence)

    # Calculate scores for all isotypes
    raw_scores = {}
    for isotype in ISOTYPES:
        raw_scores[isotype] = calculate_isotype_score(sequence, isotype)

    # Determine best isotype with tie-breaking based on motif specificity
    max_score = max(raw_scores.values())
    top_candidates = [iso for iso, score in raw_scores.items() if score == max_score]

    if len(top_candidates) == 1:
        predicted_isotype = top_candidates[0]
    else:
        sequence_upper = sequence.upper()

        def candidate_key(iso: str) -> Tuple[int, int, int]:
            matched_motifs = [len(motif) for motif, _ in CONSTANT_REGION_MOTIFS[iso] if motif in sequence_upper]
            longest_match = max(matched_motifs) if matched_motifs else 0
            match_count = len(matched_motifs)
            priority = ISOTYPE_PRIORITY.get(iso, 0)
            return (longest_match, match_count, priority)

        predicted_isotype = max(top_candidates, key=candidate_key)

    confidence = raw_scores[predicted_isotype]

    return IsotypePrediction(
        predicted_isotype=predicted_isotype,
        confidence=confidence,
        features=features,
        scores=raw_scores,  # Using raw_scores as scores for backward compatibility
        raw_scores=raw_scores
    )

def predict_isotypes(sequences: List[str]) -> List[IsotypePrediction]:  # Renamed for test compatibility
    """
    Predict isotypes for a batch of sequences.

    Args:
        sequences: List of amino acid sequence strings

    Returns:
        List of IsotypePrediction objects
    """
    return [predict_isotype(seq) for seq in sequences]
