import numpy as np
import pytest

from workspace.bioinformatics.thermal_stability_features import ThermalStabilityFeatures


def _workspace(tmp_path):
    root = tmp_path / "ws"
    (root / "data").mkdir(parents=True)
    return root


def test_sequence_thermal_features_basic(tmp_path):
    extractor = ThermalStabilityFeatures(workspace_root=_workspace(tmp_path))
    sequence = "ACDE"

    metrics = extractor.calculate_sequence_thermal_features(sequence)

    assert metrics["thermal_stability_score"] == pytest.approx(0.0)
    assert metrics["thermal_stability_sum"] == pytest.approx(0.0)
    assert metrics["thermal_stability_std"] >= 0.0
    assert metrics["cysteine_count"] == 1
    assert metrics["cysteine_density"] == pytest.approx(1 / len(sequence))
    assert metrics["helix_formers_count"] == 2


def test_abmelt_stability_score_matches_component_mean(tmp_path):
    extractor = ThermalStabilityFeatures(workspace_root=_workspace(tmp_path))
    sequence = "AAAA"

    metrics = extractor.calculate_abmelt_inspired_features(sequence)

    components = [
        min(sequence.count("C") / 4.0, 1.0),
        metrics.get("core_density", 0) * 2.0,
        metrics.get("structural_rigidity", 0) * 3.0,
        1.0 - metrics.get("flexibility_index", 1.0),
    ]

    assert metrics["abmelt_stability_score"] == pytest.approx(np.mean(components))


def test_extract_thermal_features_single_combines_scores(tmp_path):
    extractor = ThermalStabilityFeatures(workspace_root=_workspace(tmp_path))
    sequence = "ACDEFGHIKLMNPQRSTVWY"

    features = extractor.extract_thermal_features_single("AB1", sequence, sequence)

    assert features["extraction_success"]
    assert features["combined_thermal_score"] == pytest.approx(
        features["vh_thermal_stability_score"] + features["vl_thermal_stability_score"]
    )
    assert features["combined_abmelt_score"] == pytest.approx(
        features["vh_abmelt_stability_score"] + features["vl_abmelt_stability_score"]
    )
    assert features["thermal_domain_cooperativity"] == pytest.approx(
        features["vh_thermal_stability_score"] * features["vl_thermal_stability_score"]
    )
