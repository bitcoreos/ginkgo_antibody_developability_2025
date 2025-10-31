"""
Debug script for PROPERMAB training issue
"""

import sys
import os
import json

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/a0/bitcore/workspace/flab_framework/propemab')

# Import the FLAb PROPERMAB framework
from flab_propemab import FLAbPROPERMAB

# Import fragment database
from fragment_database import FragmentDatabase

def debug_propemab_training():
    """
    Debug PROPERMAB training issue.
    """
    print("=== Debugging PROPERMAB Training Issue ===\n")
    
    # Load fragment database
    print("Loading fragment database:")
    db = FragmentDatabase('test_fragment_database.json')
    fragment_database = db.load_database()
    print(f"  Fragment database type: {type(fragment_database)}")
    print(f"  Fragment database is None: {fragment_database is None}")
    
    if fragment_database is not None:
        print(f"  Number of fragments: {len(fragment_database)}")
        for fragment_id, fragment_data in fragment_database.items():
            print(f"    Fragment: {fragment_id}")
    
    # Initialize and train PROPERMAB
    print("\nInitializing and training PROPERMAB:")
    propemab = FLAbPROPERMAB()
    
    # Check if prepare_training_data works correctly
    print("  Preparing training data:")
    X, y_dict, feature_names = propemab.prepare_training_data(fragment_database)
    print(f"    X shape: {X.shape}")
    print(f"    Feature names: {feature_names}")
    
    for prop, y in y_dict.items():
        print(f"    {prop} shape: {y.shape}")
    
    # Try to train
    print("  Training PROPERMAB:")
    propemab.train(fragment_database)
    print(f"    Is trained: {propemab.is_trained}")
    
    # Try to make a prediction
    if propemab.is_trained:
        print("  Making a test prediction:")
        # Use one of the fragments from the database as test data
        if fragment_database is not None and len(fragment_database) > 0:
            test_fragment = list(fragment_database.values())[0]
            predictions = propemab.predict_developability(test_fragment)
            print(f"    Predictions: {predictions}")


def main():
    """
    Main function to run the debug script.
    """
    debug_propemab_training()


if __name__ == "__main__":
    main()
