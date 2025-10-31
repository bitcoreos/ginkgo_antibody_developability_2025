import numpy as np
import pytest

from workspace.bioinformatics.aggregation_propensity_features import AggregationPropensityFeatures


def _workspace(tmp_path):
    root = tmp_path / "ws"
    (root / "data").mkdir(parents=True)
    return root


def test_calculate_hydrophobic_surface_area_basic(tmp_path):
    extractor = AggregationPropensityFeatures(workspace_root=_workspace(tmp_path))
    sequence = "AVVL"

    metrics = extractor.calculate_hydrophobic_surface_area(sequence)

    assert metrics["hydrophobic_surface_area"] == pytest.approx(14.0)
    assert metrics["hydrophobic_ratio"] == pytest.approx(1.0)
    assert metrics["hydrophobic_density"] == pytest.approx(3.5)
    assert metrics["max_hydrophobic_patch"] == 4
    assert metrics["hydrophobic_clustering"] == pytest.approx(0.5)


def test_detect_aggregation_motifs_counts_and_density(tmp_path):
    extractor = AggregationPropensityFeatures(workspace_root=_workspace(tmp_path))
    sequence = "LLLAAAKKK"

    metrics = extractor.detect_aggregation_motifs(sequence)

    assert metrics["hydrophobic_patches_count"] == 1
    assert metrics["charge_clusters_count"] == 1
    assert metrics["hydrophobic_patches_density"] == pytest.approx(1 / 7)


def test_biophysical_risk_score_aggregates_components(tmp_path):
    extractor = AggregationPropensityFeatures(workspace_root=_workspace(tmp_path))
    sequence = "VVVKKKDDD"

    risk = extractor.calculate_biophysical_risk_scores(sequence)
    component_sum = (
        risk["hydrophobic_risk"]
        + risk["surface_risk"]
        + risk["electrostatic_risk"]
        + risk["motif_risk"]
        + risk["patch_risk"]
    )

    assert risk["aggregation_risk_score"] == pytest.approx(component_sum)


def test_extract_aggregation_features_single_combines_domains(tmp_path):
    extractor = AggregationPropensityFeatures(workspace_root=_workspace(tmp_path))

    features = extractor.extract_aggregation_features_single("AB1", "VVVV", "KKKK")

    assert features["extraction_success"]
    assert features["combined_hydrophobic_area"] == pytest.approx(
        features["vh_hydrophobic_surface_area"] + features["vl_hydrophobic_surface_area"]
    )
    assert features["combined_risk_score"] == pytest.approx(
        features["vh_aggregation_risk_score"] + features["vl_aggregation_risk_score"]
    )
    assert features["vh_vl_charge_balance"] == pytest.approx(
        abs(features["vh_net_charge"] - features["vl_net_charge"])
    )
