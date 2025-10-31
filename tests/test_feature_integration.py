import numpy as np
import pandas as pd


def _make_feature_matrix(num_rows: int = 25):
    antibody_ids = [f"AB{i:03d}" for i in range(num_rows)]
    folds = [i % 5 for i in range(num_rows)]
    feature_a = np.linspace(0.0, 1.0, num_rows)
    feature_b = np.sin(feature_a * np.pi)
    target_vals = 2.0 + feature_a * 0.5

    return pd.DataFrame(
        {
            "antibody_id": antibody_ids,
            "fold": folds,
            "feature_a": feature_a,
            "feature_b": feature_b,
            "HIC_delta_G_ML": target_vals,
        }
    )


def _make_targets(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    return feature_matrix[["antibody_id", "HIC_delta_G_ML"]].copy()


def test_create_baseline_models_excludes_targets(monkeypatch):
    from workspace.bioinformatics.feature_integration import FeatureIntegration

    feature_matrix = _make_feature_matrix()
    targets = _make_targets(feature_matrix)

    integrator = FeatureIntegration()
    integrator.target_columns = ["HIC_delta_G_ML"]

    results = integrator.create_baseline_models(feature_matrix, targets)

    assert "HIC_delta_G_ML" in results
    hic_result = results["HIC_delta_G_ML"]

    assert "error" not in hic_result
    assert hic_result["n_samples"] == len(feature_matrix)
    assert "model" in hic_result and "scaler" in hic_result
    assert "imputer" in hic_result
    assert hic_result["feature_columns"]


def test_train_xgboost_models(monkeypatch):
    from workspace.bioinformatics.feature_integration import FeatureIntegration

    feature_matrix = _make_feature_matrix(30)
    targets = _make_targets(feature_matrix)

    integrator = FeatureIntegration()
    integrator.target_columns = ["HIC_delta_G_ML"]

    results = integrator.train_xgboost_models(feature_matrix, targets)

    assert "HIC_delta_G_ML" in results
    xgb_result = results["HIC_delta_G_ML"]

    assert "error" not in xgb_result
    assert xgb_result["n_samples"] == len(feature_matrix)
    assert "model" in xgb_result
    assert "imputer" in xgb_result