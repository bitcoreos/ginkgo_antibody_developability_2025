#!/usr/bin/env python3
"""
Feature Integration and Baseline Model

Combines all evidence-based features and creates initial predictive models
for the antibody developability competition.

Features integrated:
1. CDR features (basic extraction)
2. Aggregation propensity features (r=0.91 target)  
3. Thermal stability features (Spearman 0.4-0.52 target)

Author: BITCORE Feature Engineering Team
Date: 2025-10-14
Purpose: Create competition-ready feature matrix and baseline predictions
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional
import json
from datetime import datetime
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureIntegration:
    """
    Integrate all evidence-based features and create baseline models.
    """
    
    def __init__(self, workspace_root: str = "/a0/bitcore/workspace"):
        self.workspace_root = Path(workspace_root)
        self.data_dir = self.workspace_root / "data"
        self.features_dir = self.data_dir / "features"
        self.models_dir = self.workspace_root / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.targets_dir = self.data_dir / "targets"
        self.targets_file = self.targets_dir / "gdpa1_competition_targets.csv"
        
        # Competition target columns (to be filled when target data is available)
        self.target_columns = [
            'HIC_delta_G_ML',
            'PR_CHO', 
            'AC-SINS_pH7.4_nmol/mg',
            'Tm2_DSF_degC',
            'Titer_g/L'
        ]
        
    def load_and_combine_features(self) -> pd.DataFrame:
        """Load all feature sets and combine them"""
        logger.info("Loading and combining all feature sets")
        
        # Load original dataset
        original_df = pd.read_csv(self.data_dir / "sequences" / "GDPa1_v1.2_sequences_processed.csv")
        logger.info(f"Loaded original dataset: {len(original_df)} antibodies")
        
        # Start with base dataset (just antibody_id for now)
        combined_df = original_df[['antibody_id']].copy()
        
        # Add fold information for CV
        combined_df['fold'] = original_df['hierarchical_cluster_IgG_isotype_stratified_fold']

        # Add hc_subtype and lc_subtype
        combined_df["hc_subtype"] = original_df["hc_subtype"]
        combined_df["lc_subtype"] = original_df["lc_subtype"]
        
        # Load CDR features
        try:
            cdr_df = pd.read_csv(self.features_dir / "cdr_features_basic.csv")
            logger.info(f"Loaded CDR features: {len(cdr_df.columns)} features")
            combined_df = combined_df.merge(cdr_df, on='antibody_id', how='left')
        except FileNotFoundError:
            logger.warning("CDR features not found")
        
        # Load aggregation features
        try:
            agg_df = pd.read_csv(self.features_dir / "aggregation_propensity_features.csv")
            logger.info(f"Loaded aggregation features: {len(agg_df.columns)} features")
            combined_df = combined_df.merge(agg_df, on='antibody_id', how='left')
        except FileNotFoundError:
            logger.warning("Aggregation features not found")
        
        # Load thermal stability features
        try:
            thermal_df = pd.read_csv(self.features_dir / "thermal_stability_features.csv")
            logger.info(f"Loaded thermal features: {len(thermal_df.columns)} features")
            combined_df = combined_df.merge(thermal_df, on='antibody_id', how='left')
        except FileNotFoundError:
            logger.warning("Thermal features not found")
        
        logger.info(f"Combined feature matrix: {len(combined_df)} samples × {len(combined_df.columns)} features")
        
        return combined_df
    
    def prepare_features_for_modeling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning with reduction to exactly 87 features"""
        logger.info("Preparing features for modeling with reduction to 87 features")

        feature_cols = self.get_feature_columns(df)

        # Remove extraction status columns
        feature_cols = [col for col in feature_cols
                       if not col.endswith('_success') and not col.endswith('_error')]

        # Handle categorical features (hc_subtype and lc_subtype)
        categorical_cols = [col for col in feature_cols if col in ['hc_subtype', 'lc_subtype']]
        if categorical_cols:
            # Apply one-hot encoding to categorical features
            encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            categorical_encoded = encoder.fit_transform(df[categorical_cols])
            categorical_feature_names = encoder.get_feature_names_out(categorical_cols)
            categorical_df = pd.DataFrame(categorical_encoded, columns=categorical_feature_names, index=df.index)

            # Select numeric features (excluding categorical, antibody_id, and fold)
            numeric_cols = [col for col in feature_cols if col not in categorical_cols and col not in ['antibody_id', 'fold']]
            # Further filter to ensure only numeric columns are selected
            X_numeric = df[numeric_cols].select_dtypes(include=[np.number])
            X = pd.concat([X_numeric, categorical_df], axis=1)
        else:
            # Get features and select only numeric columns
            X = df[feature_cols].select_dtypes(include=[np.number])

        # Apply feature reduction to get exactly 87 features
        X_reduced = self.reduce_to_87_features(X, df)

        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X_reduced),
            columns=X_reduced.columns,
            index=X_reduced.index
        )

        # Add back antibody_id
        antibody_id_series = df["antibody_id"].reset_index(drop=True)
        X_final = pd.concat([antibody_id_series, X_imputed.reset_index(drop=True)], axis=1)

        logger.info(f"Final feature matrix shape: {X_final.shape}")

        return X_final
    def reduce_to_87_features(self, X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce features to exactly 87 using mutual information or other selection criteria"""
        logger.info(f"Reducing features from {X.shape[1]} to 87")

        # Select only numeric columns to avoid TypeError when calculating variances
        X_numeric = X.select_dtypes(include=[np.number])
        logger.info(f"Selected {X_numeric.shape[1]} numeric columns from {X.shape[1]} total columns")

        # If we already have 87 or fewer numeric features, return as is
        if X_numeric.shape[1] <= 87:
            logger.info(f"Already have {X_numeric.shape[1]} numeric features, no reduction needed")
            return X_numeric

        # For now, we'll use a simple approach to select top 87 features based on variance
        # In a more advanced version, we could use mutual information with targets
        # Calculate variance for each numeric feature
        variances = X_numeric.var()
        # Sort features by variance (descending)
        sorted_features = variances.sort_values(ascending=False)
        # Select top 87 features
        selected_features = sorted_features.head(87).index.tolist()
        X_reduced = X_numeric[selected_features]

        logger.info(f"Reduced to {X_reduced.shape[1]} features")
        return X_reduced
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Return the feature column names excluding metadata and known targets."""

        excluded = {'antibody_id', 'fold'}
        excluded.update(self.target_columns)
        return [col for col in df.columns if col not in excluded]

    def load_targets(self) -> Optional[pd.DataFrame]:
        """Load cleaned target assays if available."""
        if not self.targets_file.exists():
            logger.warning("Target assay file not found: %s", self.targets_file)
            return None

        targets_df = pd.read_csv(self.targets_file)
        if 'antibody_id' not in targets_df.columns:
            raise ValueError("Target file missing antibody_id column")

        missing_columns = [col for col in self.target_columns if col not in targets_df.columns]
        if missing_columns:
            raise ValueError(f"Target file missing expected columns: {missing_columns}")

        targets_df['antibody_id'] = targets_df['antibody_id'].astype(str)
        logger.info("Loaded targets with coverage: %s", targets_df[self.target_columns].notna().sum().to_dict())
        return targets_df
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

            imputer = SimpleImputer(strategy='median')
            X_train_imputed = imputer.fit_transform(X_train)

            scaler: Optional[StandardScaler]
            if scale_features:
                scaler = StandardScaler()
                X_train_transformed = scaler.fit_transform(X_train_imputed)
            else:
                scaler = None
                X_train_transformed = X_train_imputed

            cv_scores: List[float] = []
            fold_predictions: List[Dict[str, float]] = []

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
                'n_samples': int(len(train_data)),
                'fold_predictions': fold_predictions,
                'all_predictions': all_prediction_list,  # New field with predictions for all antibodies
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

        try:
            from xgboost import XGBRegressor  # type: ignore
        except ImportError as exc:
            raise RuntimeError("xgboost is not installed; add it to requirements to enable this model") from exc

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

    def save_feature_matrix(
        self,
        combined_df: pd.DataFrame,
        feature_matrix: pd.DataFrame,
        targets_df: Optional[pd.DataFrame] = None
    ):
        """Save the combined feature matrix for future modeling"""
        logger.info("Saving feature matrices")
        
        # Save combined feature matrix with metadata
        feature_matrix_file = self.features_dir / "combined_feature_matrix.csv"
        combined_df.to_csv(feature_matrix_file, index=False)
        logger.info(f"Combined feature matrix saved: {feature_matrix_file}")
        
        # Save modeling-ready feature matrix
        modeling_matrix_file = self.features_dir / "modeling_feature_matrix.csv"
        modeling_df = feature_matrix.copy()

        target_coverage = None
        if targets_df is not None:
            modeling_df = modeling_df.merge(
                targets_df[['antibody_id'] + self.target_columns],
                on='antibody_id',
                how='left'
            )
            target_coverage = modeling_df[self.target_columns].notna().sum().to_dict()
            logger.info("Appended target assays to modeling matrix")

        modeling_df.to_csv(modeling_matrix_file, index=False)
        logger.info(f"Modeling feature matrix saved: {modeling_matrix_file}")
        
        # Save feature summary
        feature_summary = {
            'generation_time': datetime.now().isoformat(),
            'total_antibodies': len(combined_df),
            'total_features': len(combined_df.columns) - 2,  # Exclude antibody_id and fold
            'modeling_features': len(feature_matrix.columns) - 2,  # Exclude antibody_id and fold
            'feature_categories': {
                'cdr_features': len([col for col in feature_matrix.columns if 'cdr_' in col]),
                'aggregation_features': len([col for col in feature_matrix.columns if any(agg in col for agg in ['hydrophobic', 'surface', 'electrostatic', 'motif'])]),
                'thermal_features': len([col for col in feature_matrix.columns if any(thermal in col for thermal in ['thermal', 'abmelt', 'disulfide'])]),
                'vhh_features': len([col for col in feature_matrix.columns if 'vhh' in col])
            },
            'ready_for_modeling': targets_df is not None,
            'target_coverage': target_coverage,
            'next_steps': [
                'Train predictive models using 5-fold CV',
                'Generate competition predictions'
            ] if targets_df is not None else [
                'Populate target assays via bioinformatics.target_ingestion',
                'Train predictive models using 5-fold CV',
                'Generate competition predictions'
            ]
        }
        
        summary_file = self.features_dir / "feature_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(feature_summary, f, indent=2)
        
        logger.info(f"Feature summary saved: {summary_file}")

def main():
    """Main execution function"""
    logger.info("Starting feature integration")
    
    # Initialize integrator
    integrator = FeatureIntegration()
    
    # Load and combine all features
    combined_df = integrator.load_and_combine_features()
    
    # Prepare for modeling
    feature_matrix = integrator.prepare_features_for_modeling(combined_df)
    
    # Load targets if available
    targets_df = integrator.load_targets()

    # Save feature matrices
    integrator.save_feature_matrix(combined_df, feature_matrix, targets_df)
    
    # Summary
    print(f"\nFeature Integration Complete:")
    print(f"- Combined feature matrix: {combined_df.shape}")
    print(f"- Modeling feature matrix: {feature_matrix.shape}")
    print(f"- Ready for modeling: {'Yes' if targets_df is not None else 'Pending target ingestion'}")
    
    # Feature breakdown
    feature_counts = {
        'CDR features': len([col for col in feature_matrix.columns if 'cdr_' in col]),
        'Aggregation features': len([col for col in feature_matrix.columns if any(agg in col for agg in ['hydrophobic', 'surface', 'electrostatic', 'motif'])]),
        'Thermal features': len([col for col in feature_matrix.columns if any(thermal in col for thermal in ['thermal', 'abmelt', 'disulfide'])]),
        'VHH features': len([col for col in feature_matrix.columns if 'vhh' in col])
    }
    
    print(f"\nFeature Categories:")
    for category, count in feature_counts.items():
        print(f"- {category}: {count}")
    
    if targets_df is not None:
        coverage = targets_df[integrator.target_columns].notna().sum()
        print("\nTarget Coverage:")
        for target, count in coverage.items():
            print(f"- {target}: {count} measurements")
    else:
        print("\nTargets not yet ingested. Run bioinformatics/target_ingestion.py to populate assays.")

    print(f"\nNext Steps:")
    if targets_df is None:
        print(f"- Ingest GDPa1 target assays")
    print(f"- Train models using 5-fold CV")
    print(f"- Generate competition predictions")
    
    return combined_df, feature_matrix

if __name__ == "__main__":
    main()