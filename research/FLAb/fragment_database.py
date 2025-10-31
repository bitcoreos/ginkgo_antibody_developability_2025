"""
Fragment Database Implementation

This module implements fragment library database management for the FLAb framework.
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime


class FragmentDatabase:
    """
    Fragment library database management.
    """
    
    def __init__(self, database_file: str = None):
        """
        Initialize the fragment database.
        
        Args:
            database_file (str): Path to database file for persistence
        """
        self.fragments = {}
        self.database_file = database_file or '/a0/bitcore/workspace/research/FLAb/fragment_database.json'
        self._load_database()
    
    def _load_database(self):
        """
        Load database from file if it exists.
        """
        if os.path.exists(self.database_file):
            try:
                with open(self.database_file, 'r') as f:
                    data = json.load(f)
                    # Convert string keys back to proper format if needed
                    self.fragments = {str(k): v for k, v in data.items()}
            except Exception as e:
                print(f"Warning: Could not load database from {self.database_file}: {e}")
                self.fragments = {}
        else:
            self.fragments = {}
    
    def _save_database(self):
        """
        Save database to file.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.database_file), exist_ok=True)
            
            # Save database
            with open(self.database_file, 'w') as f:
                json.dump(self.fragments, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save database to {self.database_file}: {e}")
    
    def add_fragment(self, fragment_id: str, fragment_sequence: str, metadata: Dict = None) -> bool:
        """
        Add a fragment to the database.
        
        Args:
            fragment_id (str): Unique identifier for the fragment
            fragment_sequence (str): Fragment sequence
            metadata (Dict): Additional metadata about the fragment
            
        Returns:
            bool: True if fragment was added successfully
        """
        # Validate inputs
        if not fragment_id or not fragment_sequence:
            return False
        
        # Check if fragment already exists
        if fragment_id in self.fragments:
            print(f"Warning: Fragment {fragment_id} already exists in database")
            return False
        
        # Add fragment to database
        self.fragments[fragment_id] = {
            'sequence': fragment_sequence,
            'metadata': metadata or {},
            'added_date': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat()
        }
        
        # Save database
        self._save_database()
        
        return True
    
    def get_fragment(self, fragment_id: str) -> Optional[Dict]:
        """
        Retrieve a fragment from the database.
        
        Args:
            fragment_id (str): Unique identifier for the fragment
            
        Returns:
            Optional[Dict]: Fragment data or None if not found
        """
        return self.fragments.get(fragment_id)
    
    def update_fragment(self, fragment_id: str, fragment_sequence: str = None, metadata: Dict = None) -> bool:
        """
        Update a fragment in the database.
        
        Args:
            fragment_id (str): Unique identifier for the fragment
            fragment_sequence (str): Updated fragment sequence
            metadata (Dict): Updated metadata
            
        Returns:
            bool: True if fragment was updated successfully
        """
        # Check if fragment exists
        if fragment_id not in self.fragments:
            print(f"Warning: Fragment {fragment_id} not found in database")
            return False
        
        # Update fragment data
        if fragment_sequence is not None:
            self.fragments[fragment_id]['sequence'] = fragment_sequence
        
        if metadata is not None:
            # Merge metadata
            self.fragments[fragment_id]['metadata'].update(metadata)
        
        # Update last modified timestamp
        self.fragments[fragment_id]['last_modified'] = datetime.now().isoformat()
        
        # Save database
        self._save_database()
        
        return True
    
    def delete_fragment(self, fragment_id: str) -> bool:
        """
        Delete a fragment from the database.
        
        Args:
            fragment_id (str): Unique identifier for the fragment
            
        Returns:
            bool: True if fragment was deleted successfully
        """
        # Check if fragment exists
        if fragment_id not in self.fragments:
            print(f"Warning: Fragment {fragment_id} not found in database")
            return False
        
        # Delete fragment
        del self.fragments[fragment_id]
        
        # Save database
        self._save_database()
        
        return True
    
    def list_fragments(self) -> List[str]:
        """
        List all fragment IDs in the database.
        
        Returns:
            List[str]: List of fragment IDs
        """
        return list(self.fragments.keys())
    
    def search_fragments(self, query: str, search_fields: List[str] = None) -> List[str]:
        """
        Search for fragments in the database.
        
        Args:
            query (str): Search query
            search_fields (List[str]): Fields to search in (sequence, metadata keys)
            
        Returns:
            List[str]: List of matching fragment IDs
        """
        if search_fields is None:
            search_fields = ['sequence', 'metadata']
        
        matching_fragments = []
        
        for fragment_id, fragment_data in self.fragments.items():
            # Search in specified fields
            for field in search_fields:
                if field == 'sequence':
                    if query.lower() in fragment_data.get('sequence', '').lower():
                        matching_fragments.append(fragment_id)
                        break
                elif field == 'metadata':
                    # Search in metadata values
                    metadata = fragment_data.get('metadata', {})
                    for key, value in metadata.items():
                        if isinstance(value, str) and query.lower() in value.lower():
                            matching_fragments.append(fragment_id)
                            break
                        elif str(value) == query:
                            matching_fragments.append(fragment_id)
                            break
                else:
                    # Search in specific metadata key
                    metadata = fragment_data.get('metadata', {})
                    value = metadata.get(field)
                    if value is not None:
                        if isinstance(value, str) and query.lower() in value.lower():
                            matching_fragments.append(fragment_id)
                            break
                        elif str(value) == query:
                            matching_fragments.append(fragment_id)
                            break
        
        return matching_fragments
    
    def get_fragment_count(self) -> int:
        """
        Get the total number of fragments in the database.
        
        Returns:
            int: Number of fragments
        """
        return len(self.fragments)
    
    def get_database_info(self) -> Dict:
        """
        Get information about the database.
        
        Returns:
            Dict: Database information
        """
        return {
            'fragment_count': len(self.fragments),
            'database_file': self.database_file,
            'last_modified': datetime.now().isoformat()
        }


def main():
    """
    Example usage of the fragment database.
    """
    # Create database instance
    db = FragmentDatabase()
    
    # Add fragments
    db.add_fragment(
        "fragment_001", 
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVARIYYSGSTNYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS",
        {"source": "synthetic library", "type": "VH", "library": "synthetic VH panel 1"}
    )
    
    db.add_fragment(
        "fragment_002",
        "DIQMTQSPSSLSASVGDRVTITCRASQSVSSSYLAWYQQKPGKAPKLLIYDASNRATGIPDRFSGSGSGTDFTLTISSLQPEDFATYYCQQRSNWPRTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC",
        {"source": "natural library", "type": "VL", "library": "human VL repertoire"}
    )
    
    # Retrieve fragment
    fragment = db.get_fragment("fragment_001")
    print("Retrieved Fragment:")
    print(f"  ID: fragment_001")
    print(f"  Sequence: {fragment['sequence']}")
    print(f"  Metadata: {fragment['metadata']}")
    print(f"  Added Date: {fragment['added_date']}")
    
    # Update fragment
    db.update_fragment("fragment_001", metadata={"expression_level": "high", "solubility": "good"})
    
    # Search fragments
    vh_fragments = db.search_fragments("VH", ["metadata"])
    print(f"\nVH Fragments: {vh_fragments}")
    
    # List all fragments
    all_fragments = db.list_fragments()
    print(f"\nAll Fragments: {all_fragments}")
    
    # Database info
    db_info = db.get_database_info()
    print(f"\nDatabase Info:")
    print(f"  Fragment Count: {db_info['fragment_count']}")
    print(f"  Database File: {db_info['database_file']}")


if __name__ == "__main__":
    main()
