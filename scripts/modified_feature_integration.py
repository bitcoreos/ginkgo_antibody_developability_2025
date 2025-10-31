"""
Modified FeatureIntegration class that handles missing target values without filtering samples
or using placeholder values.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Callable, Dict, List, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

# Try to import XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBRegressor = Any  # Just for type checking

logger = logging.getLogger(__name__)

class ModifiedFeatureIntegration:
    """
    Modified FeatureIntegration class that handles missing target values without filtering samples
    or using placeholder values.
    """

    def __init__(self, workspace_root: str = "/a0/bitcore/workspace"):
        self.workspace_root = Path(workspace_root)
        self.data_dir = self.workspace_root / "data"
        self.features_dir = self.data_dir / "features"
        self.models_dir = self.workspace_root / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.targets_dir = self.data_dir / "targets"
        self.targets_file = self.targets_dir / "gdpa1_competition_targets.csv"

        # Competition target columns
        self.target_columns = [
            'HIC_delta_G_ML',
            'PR_CHO', 
            'AC-SINS_pH7.4_nmol/mg',
            'Tm2_DSF_degC',
            'Titer_g/L'
        ]

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Return the feature column names excluding metadata and known targets."""
        excluded = {'antibody_id', 'fold'}
        excluded.update(self.target_columns)
        return [col for col in df.columns if col not in excluded]

    def _train_models(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        model_factory: Callable[[], object],
        model_name: str,
        scale_features: bool = True,
    ) -> Dict[str, Dict]:
        """Generic helper to train per-target models using shared feature matrix."""

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

        results: Dict[str, Dict] = {}

        for target in available_targets:
            logger.info("[%s] Training model for %s", model_name, target)

            # Merge feature frame with target data
            merged = feature_frame.merge(target_lookup[['antibody_id', target]], on='antibody_id', how='left')

            # Use all samples for both training and prediction (don't filter out missing targets)
            # This is the key change from the original implementation
            all_data = merged  # All antibodies with valid features

            if all_data.empty:
                logger.warning("[%s] No data available for %s; skipping", model_name, target)
                results[target] = {'error': 'no_data'}
                continue

            # Process data
            if 'fold' in all_data.columns:
                fold_series = all_data['fold']
                if fold_series.isna().any():
                    logger.warning("[%s] Missing fold assignments for %s; assigning fallback fold -1", model_name, target)
                    fold_series = fold_series.fillna(-1)
                folds = fold_series.astype(int).to_numpy()
            else:
                folds = np.zeros(len(all_data), dtype=int)

            X_all = all_data[feature_cols]
            y_all = all_data[target]  # This will contain NaN values for missing targets

            # Impute only feature values, not target values
            imputer = SimpleImputer(strategy='median')
            X_all_imputed = imputer.fit_transform(X_all)

            scaler: Optional[StandardScaler]
            if scale_features:
                scaler = StandardScaler()
                X_all_transformed = scaler.fit_transform(X_all_imputed)
            else:
                scaler = None
                X_all_transformed = X_all_imputed

            cv_scores: List[float] = []
            fold_predictions: List[Dict[str, float]] = []

            unique_folds = sorted({int(f) for f in folds}) or [0]

            for fold_id in unique_folds:
                val_mask = folds == fold_id
                train_mask = ~val_mask

                if not val_mask.any() or not train_mask.any():
                    continue

                # For training, only use samples with non-missing target values
                # This is important for algorithms that cannot handle missing targets
                train_non_missing_mask = train_mask & (~y_all.isna().values)
                val_non_missing_mask = val_mask & (~y_all.isna().values)

                # Check if we have enough samples for training and validation
                if not train_non_missing_mask.any() or not val_non_missing_mask.any():
                    continue

                model = model_factory()
                train_features = X_all_transformed[train_non_missing_mask]
                train_targets = y_all.values[train_non_missing_mask]
                val_features = X_all_transformed[val_non_missing_mask]
                val_targets = y_all.values[val_non_missing_mask]

                # Train the model only on samples with non-missing targets
                model.fit(train_features, train_targets)
                preds = model.predict(val_features)

                score = r2_score(val_targets, preds)
                cv_scores.append(score)

                # Store predictions for samples with known targets
                for antibody, actual, predicted in zip(
                    all_data['antibody_id'].values[val_non_missing_mask],
                    val_targets,
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

            # Train final model on all available training data (samples with non-missing targets)
            final_model = model_factory()
            # Create mask for samples with non-missing targets
            non_missing_mask = ~y_all.isna().values

            # Check if we have any samples with non-missing targets
            if non_missing_mask.any():
                X_train_final = X_all_transformed[non_missing_mask]
                y_train_final = y_all.values[non_missing_mask]
                final_model.fit(X_train_final, y_train_final)
            else:
                logger.warning("[%s] No samples with non-missing targets for %s; model not trained", model_name, target)
                results[target] = {'error': 'no_non_missing_targets'}
                continue

            # Generate predictions for ALL antibodies (including those with missing targets)
            all_predictions = final_model.predict(X_all_transformed)

            # Create list of all predictions
            all_prediction_list = []
            for antibody, predicted in zip(
                all_data['antibody_id'].values,
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

            feature_importance: List[Dict[str, float]] = []
            if hasattr(final_model, 'feature_importances_'):
                importance = getattr(final_model, 'feature_importances_')
                feature_importance = (
                    pd.DataFrame(
                        {
                            'feature': feature_cols,
                            'importance': importance,
                        }
                    )
                    .sort_values('importance', ascending=False)
                    .head(10)
                    .to_dict('records')
                )

            results[target] = {
                'cv_scores': cv_scores,
                'mean_cv_score': float(np.mean(cv_scores)) if cv_scores else float('nan'),
                'std_cv_score': float(np.std(cv_scores)) if cv_scores else float('nan'),
                'n_samples': int(non_missing_mask.sum()),  # Number of samples with non-missing targets
                'fold_predictions': fold_predictions,
                'all_predictions': all_prediction_list,  # Predictions for all antibodies
                'top_features': feature_importance,
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

    def create_baseline_models(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict:
        """Create baseline random forest models for each target."""

        def _rf_factory() -> RandomForestRegressor:
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                random_state=42,
                n_jobs=-1,
            )

        return self._train_models(X, y, _rf_factory, model_name="random_forest", scale_features=True)

    def train_xgboost_models(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict:
        """Train gradient-boosted decision tree models (XGBoost) for each target."""
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("xgboost is not installed; add it to requirements to enable this model")

        def _xgb_factory() -> XGBRegressor:
            return XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                reg_lambda=1.0,
                reg_alpha=0.0,
                objective='reg:squarederror',
                verbosity=0,
            )

        return self._train_models(X, y, _xgb_factory, model_name="xgboost", scale_features=True)

    def generate_cv_predictions(self, results: Dict) -> pd.DataFrame:
        """Generate cross-validation predictions for competition submission"""
        logger.info("Generating CV predictions for competition submission")

        all_predictions = []

        for target, result in results.items():
            # Use 'all_predictions' to include all antibodies
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
