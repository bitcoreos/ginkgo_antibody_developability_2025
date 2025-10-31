
"""
Test script for FragmentDatabase class
"""

import json
import os
from fragment_database import FragmentDatabase


def test_fragment_database():
    """
    Test the FragmentDatabase class functionality
    """
    print("Testing FragmentDatabase:")

    # Create a database instance
    db = FragmentDatabase('test_fragment_database.json')

    # Add a test fragment
    test_fragment_id = "test_fragment_001"
    test_fragment_data = {
        "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDFWSGYWGQGTLVTVSS",
        "analysis": {
            "length": 117,
            "composition": {
                "hydrophobic": 65,
                "charged": 20,
                "polar": 32,
                "aromatic": 18
            },
            "physicochemical_properties": {
                "net_charge": 2,
                "hydrophobicity": -0.26,
                "isoelectric_point": 7.88
            },
            "stability": {
                "thermal_stability": 0.58,
                "aggregation_propensity": 0.57
            }
        }
    }

    # Add fragment to database
    db.add_fragment(test_fragment_id, test_fragment_data)

    # Retrieve fragment from database
    retrieved_fragment = db.get_fragment(test_fragment_id)
    print(f"Retrieved fragment: {retrieved_fragment is not None}")

    # Search fragments
    search_results = db.search_fragments({'analysis.stability.thermal_stability': 0.58})
    print(f"Search results: {search_results}")

    # Get database info
    db_info = db.get_database_info()
    print(f"Database info: {db_info}")

    # Test backup functionality
    print("Testing backup and restore functionality:")
    backup_path = db.backup_database()
    print(f"Backup path: {backup_path}")

    if backup_path and os.path.exists(backup_path):
        print("Backup successful")

        # Test restore functionality
        restore_result = db.restore_database(backup_path)
        print(f"Restore result: {restore_result}")
    else:
        print("Backup failed")

    # Clean up test database file
    if os.path.exists('test_fragment_database.json'):
        os.remove('test_fragment_database.json')
        print("Cleaned up test database file.")

    # Clean up backup file
    if backup_path and os.path.exists(backup_path):
        os.remove(backup_path)
        print("Cleaned up backup file.")

if __name__ == "__main__":
    test_fragment_database()
