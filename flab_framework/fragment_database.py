"""
Fragment Database Module for FLAb Framework

This module provides persistent storage and search functionality for antibody fragments.
"""

import json
import os
from datetime import datetime


class FragmentDatabase:
    """
    Class for persistent storage and search functionality for antibody fragments.
    """

    def __init__(self, database_file='fragment_database.json'):
        """
        Initialize the FragmentDatabase.

        Args:
        database_file (str): Path to the JSON database file
        """
        self.database_file = database_file
        self.fragments = {}
        self.load_database()

    def load_database(self):
        """
        Load the fragment database from a JSON file.

        Returns:
        dict: Loaded fragment database
        """
        if os.path.exists(self.database_file):
            try:
                with open(self.database_file, 'r') as f:
                    self.fragments = json.load(f)
                print(f"Loaded {len(self.fragments)} fragments from database.")
            except Exception as e:
                print(f"Error loading database: {e}")
                self.fragments = {}
        else:
            print("Database file not found. Starting with empty database.")
            self.fragments = {}

        # Return the loaded fragments
        return self.fragments

    def save_database(self):
        """
        Save the fragment database to a JSON file.
        """
        try:
            with open(self.database_file, 'w') as f:
                json.dump(self.fragments, f, indent=2)
            print(f"Saved {len(self.fragments)} fragments to database.")
        except Exception as e:
            print(f"Error saving database: {e}")

    def add_fragment(self, fragment_id, fragment_data):
        """
        Add a fragment to the database.

        Args:
        fragment_id (str): Unique identifier for the fragment
        fragment_data (dict): Fragment data including sequence and analysis results
        """
        # Add timestamp
        fragment_data['timestamp'] = datetime.now().isoformat()

        # Add fragment to database
        self.fragments[fragment_id] = fragment_data

        # Save database
        self.save_database()

        print(f"Added fragment {fragment_id} to database.")

    def get_fragment(self, fragment_id):
        """
        Retrieve a fragment from the database.

        Args:
        fragment_id (str): Unique identifier for the fragment

        Returns:
        dict: Fragment data or None if not found
        """
        return self.fragments.get(fragment_id)

    def search_fragments(self, criteria=None):
        """
        Search for fragments based on criteria.

        Args:
        criteria (dict): Search criteria (e.g., {'property': 'value'})

        Returns:
        list: List of fragment IDs matching the criteria
        """
        if criteria is None:
            return list(self.fragments.keys())

        matching_fragments = []

        for fragment_id, fragment_data in self.fragments.items():
            match = True

            for key, value in criteria.items():
                # Handle nested keys (e.g., 'analysis.composition.hydrophobic')
                if '.' in key:
                    keys = key.split('.')
                    nested_value = fragment_data
                    try:
                        for k in keys:
                            nested_value = nested_value[k]
                            if nested_value != value:
                                match = False
                                break
                    except KeyError:
                        match = False
                        break
                else:
                    # Handle top-level keys
                    if fragment_data.get(key) != value:
                        match = False
                        break

            if match:
                matching_fragments.append(fragment_id)

        return matching_fragments

    def delete_fragment(self, fragment_id):
        """
        Delete a fragment from the database.

        Args:
        fragment_id (str): Unique identifier for the fragment
        """
        if fragment_id in self.fragments:
            del self.fragments[fragment_id]
            self.save_database()
            print(f"Deleted fragment {fragment_id} from database.")
        else:
            print(f"Fragment {fragment_id} not found in database.")

    def get_database_info(self):
        """
        Get information about the database.

        Returns:
        dict: Database information
        """
        return {
        'total_fragments': len(self.fragments),
        'database_file': self.database_file
        }


    def backup_database(self, backup_dir="backups"):
        """
        Create a backup of the database file.

        Args:
            backup_dir (str): Directory to store backups

        Returns:
            str: Path to the backup file
        """
        # Create backup directory if it doesn't exist
        os.makedirs(backup_dir, exist_ok=True)

        # Create timestamped backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{os.path.splitext(os.path.basename(self.database_file))[0]}_backup_{timestamp}.json"
        backup_path = os.path.join(backup_dir, backup_filename)

        # Copy database file to backup location
        try:
            import shutil
            shutil.copy2(self.database_file, backup_path)
            print(f"Database backed up to: {backup_path}")
            return backup_path
        except Exception as e:
            print(f"Error creating backup: {e}")
            return None

    def restore_database(self, backup_file):
        """
        Restore the database from a backup file.

        Args:
            backup_file (str): Path to the backup file

        Returns:
            bool: True if restore successful, False otherwise
        """
        try:
            import shutil
            shutil.copy2(backup_file, self.database_file)
            self.load_database()
            print(f"Database restored from: {backup_file}")
            return True
        except Exception as e:
            print(f"Error restoring database: {e}")
            return False

if __name__ == "__main__":
    # Basic test of the FragmentDatabase class
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

    # Clean up test database file
    if os.path.exists('test_fragment_database.json'):
        os.remove('test_fragment_database.json')
    print("Cleaned up test database file.")
