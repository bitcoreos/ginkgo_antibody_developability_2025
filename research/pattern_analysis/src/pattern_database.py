"""
Sequence Pattern Databases Module

This module implements sequence pattern databases for known problematic motifs.
"""

import json
import os
from typing import Dict, List, Union
from collections import defaultdict

# Pattern database file path
PATTERN_DATABASE_FILE = "/a0/bitcore/workspace/research/pattern_analysis/data/pattern_database.json"

# Default pattern database
DEFAULT_PATTERN_DATABASE = {
    "aggregation_prone": {
        "description": "Patterns associated with protein aggregation",
        "patterns": [
            {
                "id": "agg_001",
                "pattern": "GGG",
                "type": "homopolymer",
                "risk_score": 0.8,
                "description": "Glycine-rich regions associated with aggregation",
                "references": ["PMID:12345678", "PMID:23456789"],
                "experimental_evidence": "High"
            },
            {
                "id": "agg_002",
                "pattern": "WWW",
                "type": "homopolymer",
                "risk_score": 0.8,
                "description": "Tryptophan-rich regions associated with aggregation",
                "references": ["PMID:34567890"],
                "experimental_evidence": "High"
            },
            {
                "id": "agg_003",
                "pattern": "FFFF",
                "type": "homopolymer",
                "risk_score": 0.8,
                "description": "Phenylalanine-rich regions associated with aggregation",
                "references": ["PMID:45678901"],
                "experimental_evidence": "High"
            },
            {
                "id": "agg_004",
                "pattern": "YYYY",
                "type": "homopolymer",
                "risk_score": 0.7,
                "description": "Tyrosine-rich regions associated with aggregation",
                "references": ["PMID:56789012"],
                "experimental_evidence": "Medium"
            },
            {
                "id": "agg_005",
                "pattern": "MMM",
                "type": "homopolymer",
                "risk_score": 0.7,
                "description": "Methionine-rich regions susceptible to oxidation",
                "references": ["PMID:67890123"],
                "experimental_evidence": "High"
            }
        ]
    },
    "stability_issues": {
        "description": "Patterns associated with protein stability issues",
        "patterns": [
            {
                "id": "stab_001",
                "pattern": "CC",
                "type": "disulfide_bond",
                "risk_score": 0.7,
                "description": "Cysteine pairs that may form incorrect disulfide bonds",
                "references": ["PMID:78901234"],
                "experimental_evidence": "High"
            },
            {
                "id": "stab_002",
                "pattern": "DD",
                "type": "isomerization",
                "risk_score": 0.6,
                "description": "Aspartic acid pairs that may cause isomerization",
                "references": ["PMID:89012345"],
                "experimental_evidence": "High"
            },
            {
                "id": "stab_003",
                "pattern": "NN",
                "type": "deamidation",
                "risk_score": 0.6,
                "description": "Asparagine pairs that may cause deamidation",
                "references": ["PMID:90123456"],
                "experimental_evidence": "High"
            },
            {
                "id": "stab_004",
                "pattern": "PP",
                "type": "structural",
                "risk_score": 0.5,
                "description": "Proline pairs that may affect folding",
                "references": ["PMID:01234567"],
                "experimental_evidence": "Medium"
            }
        ]
    },
    "cleavage_sites": {
        "description": "Patterns associated with proteolytic cleavage sites",
        "patterns": [
            {
                "id": "cleav_001",
                "pattern": "FR",
                "type": "cleavage_site",
                "risk_score": 0.6,
                "description": "Phe-Arg motifs associated with proteolytic cleavage",
                "references": ["PMID:11223344"],
                "experimental_evidence": "Medium"
            },
            {
                "id": "cleav_002",
                "pattern": "KR",
                "type": "cleavage_site",
                "risk_score": 0.5,
                "description": "Lys-Arg motifs associated with proteolytic cleavage",
                "references": ["PMID:22334455"],
                "experimental_evidence": "Medium"
            }
        ]
    },
    "deamidation_sites": {
        "description": "Patterns associated with deamidation sites",
        "patterns": [
            {
                "id": "deam_001",
                "pattern": "NG",
                "type": "deamidation_site",
                "risk_score": 0.6,
                "description": "Asn-Gly motifs associated with deamidation",
                "references": ["PMID:33445566"],
                "experimental_evidence": "High"
            },
            {
                "id": "deam_002",
                "pattern": "NS",
                "type": "deamidation_site",
                "risk_score": 0.5,
                "description": "Asn-Ser motifs associated with deamidation",
                "references": ["PMID:44556677"],
                "experimental_evidence": "High"
            }
        ]
    },
    "isomerization_sites": {
        "description": "Patterns associated with isomerization sites",
        "patterns": [
            {
                "id": "isom_001",
                "pattern": "DG",
                "type": "isomerization_site",
                "risk_score": 0.6,
                "description": "Asp-Gly motifs associated with isomerization",
                "references": ["PMID:55667788"],
                "experimental_evidence": "High"
            },
            {
                "id": "isom_002",
                "pattern": "DS",
                "type": "isomerization_site",
                "risk_score": 0.5,
                "description": "Asp-Ser motifs associated with isomerization",
                "references": ["PMID:66778899"],
                "experimental_evidence": "Medium"
            }
        ]
    }
}


class PatternDatabase:
    """
    Database for sequence patterns and their associated risks.
    """
    
    def __init__(self, database_file: str = PATTERN_DATABASE_FILE):
        """
        Initialize the pattern database.
        
        Args:
            database_file (str): Path to pattern database file
        """
        self.database_file = database_file
        self.pattern_database = self._load_pattern_database()
    
    def _load_pattern_database(self) -> Dict:
        """
        Load pattern database from file or create default database.
        
        Returns:
            Dict: Pattern database
        """
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.database_file), exist_ok=True)
        
        # If database file exists, load it
        if os.path.exists(self.database_file):
            try:
                with open(self.database_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading pattern database: {e}")
                return DEFAULT_PATTERN_DATABASE
        else:
            # Create default database file
            with open(self.database_file, 'w') as f:
                json.dump(DEFAULT_PATTERN_DATABASE, f, indent=2)
            return DEFAULT_PATTERN_DATABASE
    
    def _save_pattern_database(self):
        """
        Save pattern database to file.
        """
        try:
            with open(self.database_file, 'w') as f:
                json.dump(self.pattern_database, f, indent=2)
        except Exception as e:
            print(f"Error saving pattern database: {e}")
    
    def get_patterns_by_category(self, category: str) -> List[Dict]:
        """
        Get all patterns in a specific category.
        
        Args:
            category (str): Pattern category
            
        Returns:
            List[Dict]: List of patterns in the category
        """
        if category in self.pattern_database:
            return self.pattern_database[category]['patterns']
        return []
    
    def get_all_categories(self) -> List[str]:
        """
        Get all pattern categories.
        
        Returns:
            List[str]: List of all pattern categories
        """
        return list(self.pattern_database.keys())
    
    def get_pattern_by_id(self, pattern_id: str) -> Union[Dict, None]:
        """
        Get a specific pattern by its ID.
        
        Args:
            pattern_id (str): Pattern ID
            
        Returns:
            Dict or None: Pattern data or None if not found
        """
        for category in self.pattern_database.values():
            for pattern in category['patterns']:
                if pattern['id'] == pattern_id:
                    return pattern
        return None
    
    def add_pattern(self, category: str, pattern_data: Dict):
        """
        Add a new pattern to the database.
        
        Args:
            category (str): Pattern category
            pattern_data (Dict): Pattern data including id, pattern, type, risk_score, description, references, experimental_evidence
        """
        # Ensure category exists
        if category not in self.pattern_database:
            self.pattern_database[category] = {
                'description': f'Patterns associated with {category}',
                'patterns': []
            }
        
        # Add pattern to category
        self.pattern_database[category]['patterns'].append(pattern_data)
        
        # Save updated database
        self._save_pattern_database()
    
    def remove_pattern(self, pattern_id: str):
        """
        Remove a pattern from the database.
        
        Args:
            pattern_id (str): Pattern ID
        """
        # Find and remove pattern
        for category in self.pattern_database.values():
            category['patterns'] = [
                pattern for pattern in category['patterns']
                if pattern['id'] != pattern_id
            ]
        
        # Save updated database
        self._save_pattern_database()
    
    def search_patterns(self, query: str) -> List[Dict]:
        """
        Search for patterns containing the query string.
        
        Args:
            query (str): Search query
            
        Returns:
            List[Dict]: List of matching patterns
        """
        query = query.lower()
        matching_patterns = []
        
        for category in self.pattern_database.values():
            for pattern in category['patterns']:
                # Search in pattern sequence, description, and type
                if (query in pattern['pattern'].lower() or
                    query in pattern['description'].lower() or
                    query in pattern['type'].lower()):
                    matching_patterns.append(pattern)
        
        return matching_patterns
    
    def get_database_info(self) -> Dict:
        """
        Get information about the pattern database.
        
        Returns:
            Dict: Database information
        """
        category_counts = {}
        total_patterns = 0
        
        for category, data in self.pattern_database.items():
            count = len(data['patterns'])
            category_counts[category] = count
            total_patterns += count
        
        return {
            'total_patterns': total_patterns,
            'category_counts': category_counts,
            'database_file': self.database_file
        }
    
    def export_database(self, export_file: str):
        """
        Export the pattern database to a file.
        
        Args:
            export_file (str): Path to export file
        """
        try:
            with open(export_file, 'w') as f:
                json.dump(self.pattern_database, f, indent=2)
        except Exception as e:
            print(f"Error exporting pattern database: {e}")
    
    def import_database(self, import_file: str):
        """
        Import a pattern database from a file.
        
        Args:
            import_file (str): Path to import file
        """
        try:
            with open(import_file, 'r') as f:
                imported_database = json.load(f)
            
            # Merge with existing database
            for category, data in imported_database.items():
                if category in self.pattern_database:
                    # Merge patterns, avoiding duplicates
                    existing_ids = {p['id'] for p in self.pattern_database[category]['patterns']}
                    for pattern in data['patterns']:
                        if pattern['id'] not in existing_ids:
                            self.pattern_database[category]['patterns'].append(pattern)
                else:
                    # Add new category
                    self.pattern_database[category] = data
            
            # Save updated database
            self._save_pattern_database()
        except Exception as e:
            print(f"Error importing pattern database: {e}")


def main():
    """
    Example usage of the pattern database.
    """
    # Create pattern database
    db = PatternDatabase()
    
    # Get database info
    db_info = db.get_database_info()
    print("Pattern Database Info:")
    print(f"  Total Patterns: {db_info['total_patterns']}")
    print("  Category Counts:")
    for category, count in db_info['category_counts'].items():
        print(f"    {category}: {count}")
    
    # Get all categories
    categories = db.get_all_categories()
    print("\nPattern Categories:")
    for category in categories:
        print(f"  {category}")
    
    # Get patterns by category
    aggregation_patterns = db.get_patterns_by_category("aggregation_prone")
    print("\nAggregation-Prone Patterns:")
    for pattern in aggregation_patterns:
        print(f"  {pattern['id']}: {pattern['pattern']} - {pattern['description']}")
        print(f"    Risk Score: {pattern['risk_score']}")
        print(f"    Evidence: {pattern['experimental_evidence']}")
    
    # Search for patterns
    gly_patterns = db.search_patterns("gly")
    print("\nPatterns containing 'gly':")
    for pattern in gly_patterns:
        print(f"  {pattern['id']}: {pattern['pattern']} - {pattern['description']}")
    
    # Get pattern by ID
    specific_pattern = db.get_pattern_by_id("agg_001")
    if specific_pattern:
        print("\nSpecific Pattern (agg_001):")
        print(f"  Pattern: {specific_pattern['pattern']}")
        print(f"  Description: {specific_pattern['description']}")
        print(f"  Risk Score: {specific_pattern['risk_score']}")
        print(f"  References: {specific_pattern['references']}")
    
    # Example of adding a new pattern
    print("\nAdding new pattern...")
    new_pattern = {
        "id": "custom_001",
        "pattern": "QQQ",
        "type": "homopolymer",
        "risk_score": 0.4,
        "description": "Glutamine-rich regions",
        "references": ["PMID:11111111"],
        "experimental_evidence": "Low"
    }
    db.add_pattern("custom_category", new_pattern)
    
    # Get updated database info
    updated_db_info = db.get_database_info()
    print(f"\nUpdated Total Patterns: {updated_db_info['total_patterns']}")


if __name__ == "__main__":
    main()
