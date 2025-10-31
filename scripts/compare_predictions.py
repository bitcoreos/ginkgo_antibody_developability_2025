import pandas as pd
import numpy as np

def compare_predictions(old_file, new_file):
    # Load the prediction files
    old_preds = pd.read_csv(old_file)
    new_preds = pd.read_csv(new_file)
    
    # Merge the predictions on antibody_id
    merged = pd.merge(old_preds, new_preds, on='antibody_id', suffixes=('_old', '_new'))
    
    # Calculate the differences for each assay
    assays = ['HIC', 'PR_CHO', 'AC-SINS_pH7.4', 'Tm2', 'Titer']
    
    print("Comparison of predictions:")
    print("=========================")
    
    for assay in assays:
        old_col = f'{assay}_old'
        new_col = f'{assay}_new'
        
        if old_col in merged.columns and new_col in merged.columns:
            # Calculate differences
            merged[f'{assay}_diff'] = merged[new_col] - merged[old_col]
            
            # Calculate statistics
            mean_diff = merged[f'{assay}_diff'].mean()
            std_diff = merged[f'{assay}_diff'].std()
            max_diff = merged[f'{assay}_diff'].max()
            min_diff = merged[f'{assay}_diff'].min()
            
            print(f"\n{assay}:")
            print(f"  Mean difference: {mean_diff:.4f}")
            print(f"  Std difference: {std_diff:.4f}")
            print(f"  Max difference: {max_diff:.4f}")
            print(f"  Min difference: {min_diff:.4f}")
            
            # Show some example differences
            print(f"  Example differences (new - old):")
            print(merged[['antibody_id', old_col, new_col, f'{assay}_diff']].head())
    
    # Calculate overall statistics
    diff_cols = [f'{assay}_diff' for assay in assays if f'{assay}_diff' in merged.columns]
    if diff_cols:
        merged['total_diff'] = merged[diff_cols].abs().sum(axis=1)
        print(f"\nOverall statistics:")
        print(f"  Mean total absolute difference: {merged['total_diff'].mean():.4f}")
        print(f"  Std total absolute difference: {merged['total_diff'].std():.4f}")
        print(f"  Max total absolute difference: {merged['total_diff'].max():.4f}")
        print(f"  Min total absolute difference: {merged['total_diff'].min():.4f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python compare_predictions.py <old_predictions.csv> <new_predictions.csv>")
        sys.exit(1)
    
    old_file = sys.argv[1]
    new_file = sys.argv[2]
    
    compare_predictions(old_file, new_file)
