#!/usr/bin/env python3
"""End-to-end competition modeling workflow for GDPa1 assays."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import sys

WORKSPACE_PATH = Path(__file__).resolve().parents[1]
if str(WORKSPACE_PATH) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_PATH))

import numpy as np
import pandas as pd
from joblib import dump

from bioinformatics.feature_integration import FeatureIntegration


def _serialize_metrics(
    model_name: str,
    results: Dict,
    artifacts: Dict,
    targets_df: pd.DataFrame,
    timestamp: str,
    cv_path: Path,
    target_columns: Iterable[str],
    training_path: Optional[Path],
    holdout_path: Optional[Path],
    training_count: int,
    holdout_count: int,
) -> Dict:
    coverage = {
        target: int(targets_df[target].notna().sum())
        for target in target_columns
        if target in targets_df.columns
    }

    metrics: Dict[str, object] = {
        "generated_at": datetime.now().isoformat(),
        "timestamp": timestamp,
    "model_type": model_name,
    "cv_predictions_file": str(cv_path),
        "target_coverage": coverage,
        "models": {},
    }

    if training_path is not None:
        metrics["training_predictions_file"] = str(training_path)
        metrics["training_prediction_count"] = training_count
    if holdout_path is not None:
        metrics["heldout_predictions_file"] = str(holdout_path)
        metrics["heldout_prediction_count"] = holdout_count

    for target, info in results.items():
        target_metrics: Dict[str, object] = {}
        if "error" in info:
            target_metrics["error"] = info["error"]
        else:
            target_metrics.update(
                {
                    "mean_r2": info.get("mean_cv_score"),
                    "std_r2": info.get("std_cv_score"),
                    "n_samples": info.get("n_samples"),
                    "cv_scores": info.get("cv_scores", []),
                    "top_features": info.get("top_features", []),
                }
            )
        if target in artifacts:
            target_metrics["artifacts"] = artifacts[target]
        metrics["models"][target] = target_metrics

    return metrics


def _safe_artifact_name(target: str) -> str:
    return target.replace('/', '_').replace(' ', '_')


def _predict_subset(
    feature_lookup: pd.DataFrame,
    feature_cols: List[str],
    candidate_ids: Iterable[str],
    model,
    imputer,
    scaler,
) -> List[Tuple[str, float]]:
    ids = [identifier for identifier in candidate_ids if identifier in feature_lookup.index]
    if not ids:
        return []

    subset = feature_lookup.loc[ids, feature_cols]
    values = subset.values

    if imputer is not None:
        values = imputer.transform(values)
    elif np.isnan(values).any():
        mask = ~np.isnan(values).any(axis=1)
        subset = subset.loc[mask]
        values = subset.values
        if subset.empty:
            return []

    if scaler is not None:
        values = scaler.transform(values)

    predictions = model.predict(values)
    return list(zip(subset.index.tolist(), predictions))


def _generate_predictions(
    feature_matrix: pd.DataFrame,
    targets_df: pd.DataFrame,
    results: Dict,
    integrator: FeatureIntegration,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = integrator.get_feature_columns(feature_matrix)
    feature_lookup = feature_matrix.set_index('antibody_id')
    feature_lookup = feature_lookup[feature_cols]

    if targets_df.empty:
        target_lookup = pd.DataFrame(columns=['antibody_id']).set_index('antibody_id')
    else:
        target_lookup = targets_df.set_index('antibody_id')

    training_ids = set(target_lookup.index)
    all_ids = set(feature_lookup.index)
    base_holdout_ids = sorted(all_ids - training_ids)

    training_records: List[Dict[str, object]] = []
    holdout_records: List[Dict[str, object]] = []

    for target, info in results.items():
        model = info.get('model')
        if model is None:
            continue

        scaler = info.get('scaler')
        imputer = info.get('imputer')
        target_feature_cols = info.get('feature_columns', feature_cols)

        target_holdout_ids = set(base_holdout_ids)
        if target in target_lookup.columns:
            missing_ids = target_lookup.index[target_lookup[target].isna()]
            target_holdout_ids.update(missing_ids)

        holdout_pairs = _predict_subset(
            feature_lookup,
            target_feature_cols,
            sorted(target_holdout_ids),
            model,
            imputer,
            scaler,
        )
        for antibody_id, prediction in holdout_pairs:
            holdout_records.append(
                {
                    'antibody_id': antibody_id,
                    'target': target,
                    'prediction': float(prediction),
                }
            )

        if target in target_lookup.columns:
            present_ids = target_lookup.index[target_lookup[target].notna()]
        else:
            present_ids = target_lookup.index

        training_pairs = _predict_subset(
            feature_lookup,
            target_feature_cols,
            sorted(present_ids),
            model,
            imputer,
            scaler,
        )
        for antibody_id, prediction in training_pairs:
            actual_value = None
            if target in target_lookup.columns:
                actual_value = target_lookup.at[antibody_id, target]
                if pd.isna(actual_value):
                    actual_value = None
            training_records.append(
                {
                    'antibody_id': antibody_id,
                    'target': target,
                    'actual': None if actual_value is None else float(actual_value),
                    'prediction': float(prediction),
                }
            )

    training_df = pd.DataFrame(training_records)
    if not training_df.empty:
        training_df = training_df.sort_values(['target', 'antibody_id']).reset_index(drop=True)

    holdout_df = pd.DataFrame(holdout_records)
    if not holdout_df.empty:
        holdout_df = holdout_df.sort_values(['target', 'antibody_id']).reset_index(drop=True)

    return training_df, holdout_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Run competition modeling pipeline")
    parser.add_argument(
        "--skip-feature-refresh",
        action="store_true",
        help="Reuse existing saved feature matrices instead of regenerating them",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Override timestamp suffix for output artifacts",
    )
    args = parser.parse_args()

    integrator = FeatureIntegration()

    if args.skip_feature_refresh and (integrator.features_dir / "modeling_feature_matrix.csv").is_file():
        combined_df = pd.read_csv(integrator.features_dir / "combined_feature_matrix.csv")
        feature_matrix = pd.read_csv(integrator.features_dir / "modeling_feature_matrix.csv")
    else:
        combined_df = integrator.load_and_combine_features()
        feature_matrix = integrator.prepare_features_for_modeling(combined_df)
        targets_df = integrator.load_targets()
        if targets_df is None:
            raise SystemExit("Target assays are missing; populate workspace/data/targets/gdpa1_competition_targets.csv")
        integrator.save_feature_matrix(combined_df, feature_matrix, targets_df)
    
    if args.skip_feature_refresh:
        targets_df = integrator.load_targets()
    else:
        targets_df = integrator.load_targets()

    if targets_df is None:
        raise SystemExit("Target assays are missing; populate workspace/data/targets/gdpa1_competition_targets.csv")

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    model_runs: List[Tuple[str, Dict[str, Dict]]] = []

    results_rf = integrator.create_baseline_models(feature_matrix, targets_df)
    model_runs.append(("random_forest", results_rf))

    try:
        results_xgb = integrator.train_xgboost_models(feature_matrix, targets_df)
        model_runs.append(("xgboost", results_xgb))
    except RuntimeError as exc:
        print(f"Skipping XGBoost training: {exc}")

    submissions_dir = integrator.workspace_root / "data" / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    research_dir = integrator.workspace_root / "research_outputs" / "model"
    research_dir.mkdir(parents=True, exist_ok=True)

    print("Competition modeling artifacts generated:")
    print(f"- Feature matrix shape: {feature_matrix.shape}")

    for model_name, results in model_runs:
        cv_predictions = integrator.generate_cv_predictions(results)
        cv_path = submissions_dir / f"{model_name}_cv_predictions_{timestamp}.csv"
        cv_predictions.to_csv(cv_path, index=False)

        training_preds, holdout_preds = _generate_predictions(feature_matrix, targets_df, results, integrator)

        training_path = None
        if not training_preds.empty:
            training_path = submissions_dir / f"{model_name}_training_predictions_{timestamp}.csv"
            training_preds.to_csv(training_path, index=False)

        holdout_path = None
        if not holdout_preds.empty:
            holdout_path = submissions_dir / f"{model_name}_heldout_predictions_{timestamp}.csv"
            holdout_preds.to_csv(holdout_path, index=False)

        model_artifacts: Dict[str, Dict[str, str]] = {}
        for target, info in results.items():
            model_obj = info.get("model")
            if model_obj is None:
                continue
            safe_target = _safe_artifact_name(f"{model_name}_{target}")
            model_path = integrator.models_dir / f"{safe_target}_model_{timestamp}.joblib"
            dump(model_obj, model_path)

            scaler_path = None
            scaler_obj = info.get("scaler")
            if scaler_obj is not None:
                scaler_path = integrator.models_dir / f"{safe_target}_scaler_{timestamp}.joblib"
                dump(scaler_obj, scaler_path)

            imputer_path = None
            imputer_obj = info.get("imputer")
            if imputer_obj is not None:
                imputer_path = integrator.models_dir / f"{safe_target}_imputer_{timestamp}.joblib"
                dump(imputer_obj, imputer_path)

            model_artifacts[target] = {
                "model": str(model_path),
            }
            if scaler_path is not None:
                model_artifacts[target]["scaler"] = str(scaler_path)
            if imputer_path is not None:
                model_artifacts[target]["imputer"] = str(imputer_path)

            # Remove large objects before serialization / reuse
            info.pop("model", None)
            info.pop("scaler", None)
            info.pop("imputer", None)

        metrics = _serialize_metrics(
            model_name,
            results,
            model_artifacts,
            targets_df,
            timestamp,
            cv_path,
            integrator.target_columns,
            training_path,
            holdout_path,
            len(training_preds),
            len(holdout_preds),
        )

        metrics_path = research_dir / f"competition_model_metrics_{model_name}_{timestamp}.json"
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        print(f"- [{model_name}] CV predictions saved to: {cv_path}")
        print(f"- [{model_name}] Metrics saved to: {metrics_path}")
        if training_path is not None:
            print(f"- [{model_name}] Training predictions saved to: {training_path}")
        if holdout_path is not None:
            print(f"- [{model_name}] Heldout predictions saved to: {holdout_path}")
        for target, data in model_artifacts.items():
            print(f"- [{model_name}] {target} model: {data['model']}")
            if 'scaler' in data:
                print(f"- [{model_name}] {target} scaler: {data['scaler']}")
            if 'imputer' in data:
                print(f"- [{model_name}] {target} imputer: {data['imputer']}")

    missing_targets = [t for t in integrator.target_columns if t not in targets_df.columns]
    if missing_targets:
        print("\nTargets missing from dataset:")
        for target in missing_targets:
            print(f"- {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
