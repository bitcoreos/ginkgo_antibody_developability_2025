import pandas as pd
import sys
import os

# Add the workspace/scripts directory to the Python path
sys.path.append('/a0/bitcore/workspace/scripts')

from apply_biological_constraints import apply_biological_constraints

def main():
    # Check if input file is provided
    if len(sys.argv) < 2:
        print("Usage: python apply_constraints_to_predictions.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.csv', '_constrained.csv')
    
    # Load predictions
    predictions = pd.read_csv(input_file)
    
    # Apply biological constraints
    constrained_preds, modifications = apply_biological_constraints(predictions)
    
    # Save constrained predictions
    constrained_preds.to_csv(output_file, index=False)
    
    # Print summary of modifications
    print(f"Applied biological constraints to {input_file}")
    print("Modifications made:")
    for assay, count in modifications.items():
        if count > 0:
            print(f"  {assay}: {count} predictions constrained")
    
    # If no modifications were made
    if all(count == 0 for count in modifications.values()):
        print("  None - all predictions were within biological ranges")

if __name__ == "__main__":
    main()
