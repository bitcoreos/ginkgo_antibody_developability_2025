
"""
Test script to verify the fix for missing predictions in the antibody developability competition.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the research directory to the path so we can import the modified module
sys.path.insert(0, "/a0/bitcore/workspace/research/bioinformatics")

# Import required modules
from sklearn.base import BaseEstimator, RegressorMixin
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a minimal implementation of the FeatureIntegrator class for testing
class TestFeatureIntegrator:
    def __init__(self):
        # Initialize with minimal required attributes
        self.target_columns = ["AC-SINS_pH7.4_nmol/mg", "Tm2_DSF_degC"]
        self.features_dir = Path("/tmp")

    def get_feature_columns(self, X):
        """Return feature columns from the dataframe"""
        return [col for col in X.columns if col not in ["antibody_id", "fold"] + self.target_columns]

    def _train_models(self, X, y, model_factory, model_name, scale_features=True):
        """Modified _train_models method for testing"""
        logger.info("Training %s models for competition targets", model_name)

        if 'antibody_id' not in X.columns:
            raise ValueError("Feature matrix must include an 'antibody_id' column")
        if 'antibody_id' not in y.columns:
            raise ValueError("Target frame must include an 'antibody_id' column")

        available_targets = [col for col in self.target_columns if col in y.columns]
        feature_cols = [
            col
            for col in self.get_feature_columns(X)
            if col not in available_targets
        ]
        if not feature_cols:
            raise ValueError("No feature columns available after excluding metadata")
        if not available_targets:
            raise ValueError("No expected target columns found in target file")

        meta_cols = ['antibody_id'] + (['fold'] if 'fold' in X.columns else [])
        feature_frame = X[meta_cols + feature_cols].copy()
        target_lookup = y[['antibody_id'] + available_targets].copy()

        results = {}

        for target in available_targets:
            logger.info("[%s] Training model for %s", model_name, target)

            # Merge feature frame with target data
            merged = feature_frame.merge(target_lookup[['antibody_id', target]], on='antibody_id', how='left')

            # Separate training data (with complete target values) from prediction data (all valid features)
            train_data = merged.dropna(subset=[target])
            prediction_data = merged  # All antibodies with valid features

            if train_data.empty:
                logger.warning("[%s] No measurements available for %s; skipping", model_name, target)
                results[target] = {'error': 'no_target_data'}
                continue

            # Process training data
            if 'fold' in train_data.columns:
                fold_series = train_data['fold']
                if fold_series.isna().any():
                    logger.warning("[%s] Missing fold assignments for %s; assigning fallback fold -1", model_name, target)
                    fold_series = fold_series.fillna(-1)
                folds = fold_series.astype(int).to_numpy()
            else:
                folds = np.zeros(len(train_data), dtype=int)

            X_train = train_data[feature_cols]
            y_train = train_data[target].astype(float)

            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X_train_imputed = imputer.fit_transform(X_train)

            from sklearn.preprocessing import StandardScaler
            scaler = None
            if scale_features:
                scaler = StandardScaler()
                X_train_transformed = scaler.fit_transform(X_train_imputed)
            else:
                X_train_transformed = X_train_imputed

            from sklearn.metrics import r2_score
            cv_scores = []
            fold_predictions = []

            unique_folds = sorted({int(f) for f in folds}) or [0]

            for fold_id in unique_folds:
                val_mask = folds == fold_id
                train_mask = ~val_mask

                if not val_mask.any() or not train_mask.any():
                    continue

                model = model_factory()
                train_features = X_train_transformed[train_mask]
                val_features = X_train_transformed[val_mask]
                model.fit(train_features, y_train.values[train_mask])
                preds = model.predict(val_features)

                score = r2_score(y_train.values[val_mask], preds)
                cv_scores.append(score)

                for antibody, actual, predicted in zip(
                    train_data['antibody_id'].values[val_mask],
                    y_train.values[val_mask],
                    preds,
                ):
                    fold_predictions.append(
                        {
                            'antibody_id': antibody,
                            'fold': int(fold_id),
                            'actual': float(actual),
                            'predicted': float(predicted),
                        }
                    )

            # Train final model on all available training data
            final_model = model_factory()
            final_model.fit(X_train_transformed, y_train.values)

            # Generate predictions for ALL antibodies with valid features
            X_predict = prediction_data[feature_cols]
            X_predict_imputed = imputer.transform(X_predict)

            if scale_features and scaler is not None:
                X_predict_transformed = scaler.transform(X_predict_imputed)
            else:
                X_predict_transformed = X_predict_imputed

            # Generate predictions for all antibodies
            all_predictions = final_model.predict(X_predict_transformed)

            # Create list of all predictions
            all_prediction_list = []
            for antibody, predicted in zip(
                prediction_data['antibody_id'].values,
                all_predictions,
            ):
                # Get actual value if available
                actual_row = target_lookup[target_lookup['antibody_id'] == antibody]
                actual = float(actual_row[target].values[0]) if not actual_row.empty and not pd.isna(actual_row[target].values[0]) else np.nan

                all_prediction_list.append(
                    {
                        'antibody_id': antibody,
                        'actual': actual,
                        'predicted': float(predicted),
                    }
                )

            results[target] = {
                'cv_scores': cv_scores,
                'mean_cv_score': float(np.mean(cv_scores)) if cv_scores else float('nan'),
                'std_cv_score': float(np.std(cv_scores)) if cv_scores else float('nan'),
                'n_samples': int(len(train_data)),
                'fold_predictions': fold_predictions,
                'all_predictions': all_prediction_list,  # New field with predictions for all antibodies
                'model': final_model,
                'scaler': scaler,
                'imputer': imputer,
                'feature_columns': feature_cols,
            }

            if cv_scores:
                logger.info(
                    "[%s] %s: R² = %.3f ± %.3f",
                    model_name,
                    target,
                    np.mean(cv_scores),
                    np.std(cv_scores),
                )
            else:
                logger.info(
                    "[%s] %s: insufficient fold coverage for CV; model fit on full data",
                    model_name,
                    target,
                )

        return results

    def generate_cv_predictions(self, results):
        """Modified generate_cv_predictions method for testing"""
        logger.info("Generating CV predictions for competition submission")

        all_predictions = []

        for target, result in results.items():
            # Use 'all_predictions' instead of 'fold_predictions' to include all antibodies
            if 'all_predictions' in result:
                for pred in result['all_predictions']:
                    # Get fold information if available
                    fold = pred.get('fold', -1) if 'fold' in pred else -1
                    all_predictions.append({
                        'antibody_id': pred['antibody_id'],
                        'target': target,
                        'fold': fold,
                        'actual': pred['actual'],
                        'predicted': pred['predicted']
                    })
            # Fallback to fold_predictions if all_predictions is not available
            elif 'fold_predictions' in result:
                for pred in result['fold_predictions']:
                    all_predictions.append({
                        'antibody_id': pred['antibody_id'],
                        'target': target,
                        'fold': pred['fold'],
                        'actual': pred['actual'],
                        'predicted': pred['predicted']
                    })

        if not all_predictions:
            logger.warning("No predictions available; returning empty CV prediction frame")
            empty_columns = ['antibody_id'] + self.target_columns
            return pd.DataFrame(columns=empty_columns)

        pred_df = pd.DataFrame(all_predictions)

        # Pivot to competition format
        cv_pred_df = pred_df.pivot(index='antibody_id', columns='target', values='predicted')
        cv_pred_df = cv_pred_df.reset_index()

        # Ensure all targets are present
        for target in self.target_columns:
            if target not in cv_pred_df.columns:
                cv_pred_df[target] = np.nan

        # Reorder columns
        cv_pred_df = cv_pred_df[['antibody_id'] + self.target_columns]

        return cv_pred_df

# Test the modified _train_models method
def test_modified_train_models():
    print("Testing modified _train_models method...")

    # Create test data with some missing target values
    feature_data = pd.DataFrame({
        "antibody_id": [f"ab{i}" for i in range(1, 11)],
        "fold": [i % 3 for i in range(1, 11)],
        "feature1": np.random.rand(10),
        "feature2": np.random.rand(10),
    })

    target_data = pd.DataFrame({
        "antibody_id": [f"ab{i}" for i in range(1, 11)],
        "AC-SINS_pH7.4_nmol/mg": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0],
        "Tm2_DSF_degC": [60.0, np.nan, 62.0, 63.0, np.nan, 65.0, 66.0, 67.0, 68.0, 69.0],
    })

    # Create an instance of the test integrator
    integrator = TestFeatureIntegrator()

    # Create a simple model factory for testing
    def model_factory():
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=10, random_state=42)

    # Test the modified _train_models method
    try:
        results = integrator._train_models(
            feature_data, 
            target_data, 
            model_factory, 
            "test_model"
        )

        # Check that we have results for both targets
        assert "AC-SINS_pH7.4_nmol/mg" in results, "Missing results for AC-SINS_pH7.4_nmol/mg"
        assert "Tm2_DSF_degC" in results, "Missing results for Tm2_DSF_degC"

        # Check that we have all_predictions for both targets
        assert "all_predictions" in results["AC-SINS_pH7.4_nmol/mg"], "Missing all_predictions for AC-SINS_pH7.4_nmol/mg"
        assert "all_predictions" in results["Tm2_DSF_degC"], "Missing all_predictions for Tm2_DSF_degC"

        # Check that all_predictions includes all antibodies
        ac_sins_predictions = results["AC-SINS_pH7.4_nmol/mg"]["all_predictions"]
        tm2_predictions = results["Tm2_DSF_degC"]["all_predictions"]

        assert len(ac_sins_predictions) == 10, f"Expected 10 predictions for AC-SINS, got {len(ac_sins_predictions)}"
        assert len(tm2_predictions) == 10, f"Expected 10 predictions for Tm2, got {len(tm2_predictions)}"

        print("SUCCESS: Modified _train_models method works correctly")
        print(f"AC-SINS predictions: {len(ac_sins_predictions)}")
        print(f"Tm2 predictions: {len(tm2_predictions)}")

        # Print some example predictions
        print("Example AC-SINS predictions:")
        for i, pred in enumerate(ac_sins_predictions[:3]):
            print(f"  {pred['antibody_id']}: predicted={pred['predicted']:.3f}, actual={pred['actual']}")

        print("Example Tm2 predictions:")
        for i, pred in enumerate(tm2_predictions[:3]):
            print(f"  {pred['antibody_id']}: predicted={pred['predicted']:.3f}, actual={pred['actual']}")

        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test the modified generate_cv_predictions method
def test_modified_generate_cv_predictions():
    print("Testing modified generate_cv_predictions method...")

    # Create test results with all_predictions
    test_results = {
        "AC-SINS_pH7.4_nmol/mg": {
            "all_predictions": [
                {"antibody_id": f"ab{i}", "predicted": i * 0.5, "actual": i * 0.5 if i % 3 != 0 else np.nan}
                for i in range(1, 11)
            ]
        },
        "Tm2_DSF_degC": {
            "all_predictions": [
                {"antibody_id": f"ab{i}", "predicted": 60 + i, "actual": 60 + i if i % 4 != 0 else np.nan}
                for i in range(1, 11)
            ]
        }
    }

    # Create an instance of the test integrator
    integrator = TestFeatureIntegrator()

    # Test the modified generate_cv_predictions method
    try:
        cv_predictions = integrator.generate_cv_predictions(test_results)

        # Check that we have predictions for all antibodies
        assert len(cv_predictions) == 10, f"Expected 10 predictions, got {len(cv_predictions)}"

        # Check that we have the expected columns
        expected_columns = ["antibody_id", "AC-SINS_pH7.4_nmol/mg", "Tm2_DSF_degC"]
        for col in expected_columns:
            assert col in cv_predictions.columns, f"Missing column: {col}"

        print("SUCCESS: Modified generate_cv_predictions method works correctly")
        print(f"Generated predictions for {len(cv_predictions)} antibodies")
        print(f"Columns: {list(cv_predictions.columns)}")

        # Print first few predictions
        print("First few predictions:")
        print(cv_predictions.head())

        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running tests for the modified antibody developability prediction methods...")

    success1 = test_modified_train_models()
    success2 = test_modified_generate_cv_predictions()

    if success1 and success2:
        print("All tests passed successfully!")
    else:
        print("Some tests failed.")
        sys.exit(1)
