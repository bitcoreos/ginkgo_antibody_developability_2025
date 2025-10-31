import pandas as pd
from pathlib import Path

def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load feature matrix, targets, and groups"""
    # Load feature matrix
    feature_path = Path(data_dir) / "features" / "clean_modeling_feature_matrix.csv"
    feature_matrix = pd.read_csv(feature_path)
    
    # Load targets
    target_path = Path(data_dir) / "targets" / "gdpa1_competition_targets.csv"
    targets = pd.read_csv(target_path)
    
    # Load sequences for groups
    sequence_path = Path(data_dir) / "sequences" / "GDPa1_v1.2_sequences_processed.csv"
    sequences = pd.read_csv(sequence_path)
    
    # Merge data
    merged = feature_matrix.merge(targets, on="antibody_id")
    merged = merged.merge(sequences[["antibody_id", "hierarchical_cluster_IgG_isotype_stratified_fold"]], 
                         on="antibody_id")
    
    # Extract features (excluding metadata columns)
    metadata_cols = ["antibody_id", "hierarchical_cluster_IgG_isotype_stratified_fold", 
                    "AC-SINS_pH7.4_nmol/mg", "Tm2_DSF_degC"]
    feature_cols = [col for col in merged.columns if col not in metadata_cols and merged[col].dtype in ["int64", "float64", "bool"]]
    X = merged[feature_cols]
    
    # Extract targets and groups
    y = merged["AC-SINS_pH7.4_nmol/mg"]  # or "Tm2_DSF_degC"
    groups = merged["hierarchical_cluster_IgG_isotype_stratified_fold"]
    
    # Drop rows with NaN targets
    non_nan_mask = y.notnull()
    merged = merged[non_nan_mask]
    X = X[non_nan_mask]
    y = y[non_nan_mask]
    groups = groups[non_nan_mask]
    
    return X, y, groups
