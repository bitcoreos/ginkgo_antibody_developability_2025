# Sequence Pattern Databases Documentation

## Overview

This document provides detailed documentation for the Sequence Pattern Databases implementation. This module provides a comprehensive database of sequence patterns and their associated risks, organized by category with detailed annotations including risk scores, references, and experimental evidence.

## Features

1. **Comprehensive Pattern Database**: Organized database of known problematic sequence patterns
2. **Category-Based Organization**: Patterns organized by type of issue (aggregation, stability, etc.)
3. **Detailed Pattern Annotations**: Each pattern includes risk score, references, and experimental evidence
4. **Database Management**: Functions for adding, removing, and searching patterns
5. **Import/Export Functionality**: Ability to import and export pattern databases
6. **Persistent Storage**: JSON-based storage with automatic loading and saving

## Implementation Details

### PatternDatabase Class

The `PatternDatabase` class is the core of the implementation:

```python
db = PatternDatabase()
```

#### Methods

- `get_patterns_by_category(category)`: Get all patterns in a specific category
- `get_all_categories()`: Get all pattern categories
- `get_pattern_by_id(pattern_id)`: Get a specific pattern by its ID
- `add_pattern(category, pattern_data)`: Add a new pattern to the database
- `remove_pattern(pattern_id)`: Remove a pattern from the database
- `search_patterns(query)`: Search for patterns containing the query string
- `get_database_info()`: Get information about the pattern database
- `export_database(export_file)`: Export the pattern database to a file
- `import_database(import_file)`: Import a pattern database from a file

### Pattern Database Structure

The implementation includes a comprehensive database of predefined problematic patterns organized by category:

#### 1. Aggregation-Prone Patterns

- "agg_001": GGG - Glycine-rich regions associated with aggregation (risk score: 0.8)
- "agg_002": WWW - Tryptophan-rich regions associated with aggregation (risk score: 0.8)
- "agg_003": FFFF - Phenylalanine-rich regions associated with aggregation (risk score: 0.8)
- "agg_004": YYYY - Tyrosine-rich regions associated with aggregation (risk score: 0.7)
- "agg_005": MMM - Methionine-rich regions susceptible to oxidation (risk score: 0.7)

#### 2. Stability Issues

- "stab_001": CC - Cysteine pairs that may form incorrect disulfide bonds (risk score: 0.7)
- "stab_002": DD - Aspartic acid pairs that may cause isomerization (risk score: 0.6)
- "stab_003": NN - Asparagine pairs that may cause deamidation (risk score: 0.6)
- "stab_004": PP - Proline pairs that may affect folding (risk score: 0.5)

#### 3. Cleavage Sites

- "cleav_001": FR - Phe-Arg motifs associated with proteolytic cleavage (risk score: 0.6)
- "cleav_002": KR - Lys-Arg motifs associated with proteolytic cleavage (risk score: 0.5)

#### 4. Deamidation Sites

- "deam_001": NG - Asn-Gly motifs associated with deamidation (risk score: 0.6)
- "deam_002": NS - Asn-Ser motifs associated with deamidation (risk score: 0.5)

#### 5. Isomerization Sites

- "isom_001": DG - Asp-Gly motifs associated with isomerization (risk score: 0.6)
- "isom_002": DS - Asp-Ser motifs associated with isomerization (risk score: 0.5)

### Pattern Data Structure

Each pattern in the database includes the following fields:

- `id`: Unique identifier for the pattern
- `pattern`: The sequence pattern
- `type`: Type of pattern (homopolymer, disulfide_bond, etc.)
- `risk_score`: Risk score between 0 and 1
- `description`: Description of the pattern and its effects
- `references`: List of relevant references (PMIDs)
- `experimental_evidence`: Level of experimental evidence (High, Medium, Low)

### Database Management

The implementation provides comprehensive database management capabilities:

1. **Adding Patterns**: New patterns can be added to existing or new categories
2. **Removing Patterns**: Patterns can be removed by ID
3. **Searching Patterns**: Patterns can be searched by sequence, description, or type
4. **Importing/Exporting**: Databases can be imported from and exported to JSON files
5. **Persistent Storage**: The database is automatically saved to and loaded from a JSON file

## Usage Example

```python
from src.pattern_database import PatternDatabase

# Create pattern database
db = PatternDatabase()

# Get database info
db_info = db.get_database_info()
print(f"Total Patterns: {db_info['total_patterns']}")

# Get patterns by category
aggregation_patterns = db.get_patterns_by_category("aggregation_prone")
for pattern in aggregation_patterns:
    print(f"{pattern['id']}: {pattern['pattern']} - {pattern['description']}")

# Search for patterns
 gly_patterns = db.search_patterns("gly")
for pattern in gly_patterns:
    print(f"{pattern['id']}: {pattern['pattern']} - {pattern['description']}")

# Add a new pattern
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
```

## Integration with FLAb Framework

This implementation can be integrated with the existing FLAb framework by:

1. Using the pattern database in the PatternRecognizer for more comprehensive pattern identification
2. Incorporating pattern database information into the MotifScorer for more detailed risk scoring
3. Using pattern database information in the OptimizationRecommender for more targeted optimization suggestions

## Future Enhancements

1. **Machine Learning Models**: Training models to predict pattern risk scores
2. **Structural Data Integration**: Integration with structural data for more accurate pattern identification
3. **Experimental Validation**: Experimental validation of identified patterns
4. **Pattern Database Expansion**: Expansion of the pattern database with new findings
5. **Context-Aware Scoring**: Adjusting pattern risk scores based on sequence context
6. **Pattern Relationship Modeling**: Modeling relationships between different patterns
