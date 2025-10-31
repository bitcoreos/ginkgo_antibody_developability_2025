import pandas as pd
import os
from pathlib import Path

def update_ledger():
    """Update the submission ledger with our new submission."""
    print("Updating submission ledger with our new submission...")
    
    # Define paths
    workspace_root = Path("/a0/bitcore/workspace")
    submissions_dir = workspace_root / "data" / "submissions"
    
    # Ledger file
    ledger_file = submissions_dir / "ledger.csv"
    
    # Our submission files
    submission_file = submissions_dir / "gdpa1_cross_validation_competition_submission_reduced_features.csv"
    manifest_file = submissions_dir / "gdpa1_cross_validation_competition_submission_reduced_features_manifest.json"
    
    # Check if the ledger file exists
    if ledger_file.exists():
        # Load existing ledger
        ledger_df = pd.read_csv(ledger_file)
        print(f"Loaded existing ledger with {len(ledger_df)} entries")
    else:
        # Create new ledger with appropriate columns
        ledger_df = pd.DataFrame(columns=[
            'timestamp',
            'submission_file',
            'submission_hash',
            'manifest_file',
            'response_id'  # This will be filled when we upload to the leaderboard
        ])
        print("Created new ledger file")
    
    # Load our manifest to get the submission hash
    import json
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    
    submission_hash = manifest['submission_hash']
    timestamp = manifest['timestamp']
    
    # Check if this submission is already in the ledger
    if submission_hash in ledger_df['submission_hash'].values:
        print(f"Submission with hash {submission_hash} already exists in ledger")
        return True
    
    # Add our submission to the ledger
    new_entry = {
        'timestamp': timestamp,
        'submission_file': str(submission_file),
        'submission_hash': submission_hash,
        'manifest_file': str(manifest_file),
        'response_id': ''  # This will be filled when we upload to the leaderboard
    }
    
    ledger_df = pd.concat([ledger_df, pd.DataFrame([new_entry])], ignore_index=True)
    
    # Save the updated ledger
    ledger_df.to_csv(ledger_file, index=False)
    print(f"Updated ledger saved to {ledger_file}")
    
    # Display the ledger
    print("Ledger contents:")
    print(ledger_df)
    
    return True

if __name__ == "__main__":
    success = update_ledger()
    if success:
        print("Submission ledger updated successfully!")
    else:
        print("Failed to update submission ledger.")
