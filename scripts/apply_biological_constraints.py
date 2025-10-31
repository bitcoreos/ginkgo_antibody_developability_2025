
import pandas as pd
import numpy as np

# Define biological constraints for each assay
BIOLOGICAL_CONSTRAINTS = {
    'HIC': {'min': 2.0, 'max': 5.0, 'description': 'Hydrophobicity (HIC) must be between 2.0-5.0'},
    'PR_CHO': {'min': 0.0, 'max': 1.0, 'description': 'Polyreactivity (PR_CHO) must be between 0.0-1.0'},
    'AC-SINS_pH7.4': {'min': -20.0, 'max': 30.0, 'description': 'Self-association (AC-SINS_pH7.4) must be between -20.0-30.0'},
    'Tm2': {'min': 60.0, 'max': 90.0, 'description': 'Thermostability (Tm2) must be between 60-90°C'},
    'Titer': {'min': 100.0, 'max': 600.0, 'description': 'Expression yield (Titer) must be between 100-600 mg/L'}
}

def apply_biological_constraints(predictions_df, constraints=BIOLOGICAL_CONSTRAINTS):
    """
    Apply biological constraints to predictions DataFrame.

    Parameters:
    predictions_df (pd.DataFrame): DataFrame with prediction columns
    constraints (dict): Dictionary defining min/max values for each assay

    Returns:
    pd.DataFrame: Constrained predictions
    """
    constrained_df = predictions_df.copy()

    # Track which predictions were modified
    modified_count = {}

    for assay, bounds in constraints.items():
        if assay in constrained_df.columns:
            # Count how many predictions are out of bounds
            out_of_bounds = ((constrained_df[assay] < bounds['min']) | 
                          (constrained_df[assay] > bounds['max']))
            modified_count[assay] = out_of_bounds.sum()

            # Apply constraints using clipping
            constrained_df[assay] = constrained_df[assay].clip(
                lower=bounds['min'], 
                upper=bounds['max']
            )

            # Log warning if any predictions were modified
            if modified_count[assay] > 0:
                print(f"Warning: {modified_count[assay]} predictions for {assay} were outside biological range and have been constrained.")

    return constrained_df, modified_count

# Example usage:
# predictions = pd.read_csv('your_predictions.csv')
# constrained_preds, modifications = apply_biological_constraints(predictions)
# constrained_preds.to_csv('constrained_predictions.csv', index=False)
