"""
Amino Acid Composition Analysis Module

This module provides functionality for analyzing amino acid composition in protein sequences.
"""

# Standard amino acids
STANDARD_AMINO_ACIDS = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
]

# Hydrophobic amino acids
HYDROPHOBIC_AMINO_ACIDS = ['A', 'F', 'G', 'I', 'L', 'M', 'P', 'V', 'W', 'Y']

# Charged amino acids
CHARGED_AMINO_ACIDS = ['R', 'H', 'K', 'D', 'E']

# Polar amino acids
POLAR_AMINO_ACIDS = ['N', 'Q', 'S', 'T', 'C']

# Aromatic amino acids
AROMATIC_AMINO_ACIDS = ['F', 'W', 'Y']

def analyze_amino_acid_composition(sequence):
    """
    Analyze amino acid composition of a protein sequence.
    
    Args:
        sequence (str): Protein sequence
        
    Returns:
        dict: Composition analysis results including:
            - counts: Dictionary of amino acid counts
            - percentages: Dictionary of amino acid percentages
            - groups: Composition by amino acid groups
    """
    # Convert sequence to uppercase
    sequence = sequence.upper()
    
    # Initialize counts
    counts = {aa: 0 for aa in STANDARD_AMINO_ACIDS}
    
    # Count amino acids
    total_aa = 0
    for aa in sequence:
        if aa in counts:
            counts[aa] += 1
            total_aa += 1
    
    # Calculate percentages
    percentages = {}
    for aa, count in counts.items():
        percentages[aa] = (count / total_aa) * 100 if total_aa > 0 else 0
    
    # Group analysis
    groups = {
        'hydrophobic': sum(counts[aa] for aa in HYDROPHOBIC_AMINO_ACIDS),
        'charged': sum(counts[aa] for aa in CHARGED_AMINO_ACIDS),
        'polar': sum(counts[aa] for aa in POLAR_AMINO_ACIDS),
        'aromatic': sum(counts[aa] for aa in AROMATIC_AMINO_ACIDS)
    }
    
    # Calculate group percentages
    group_percentages = {}
    for group, count in groups.items():
        group_percentages[group] = (count / total_aa) * 100 if total_aa > 0 else 0
    
    return {
        'counts': counts,
        'percentages': percentages,
        'total_amino_acids': total_aa,
        'groups': groups,
        'group_percentages': group_percentages
    }

if __name__ == "__main__":
    # Test the amino acid composition analysis
    test_sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDFWSGYWGQGTLVTVSS"
    
    print("Testing amino acid composition analysis:")
    print(f"Test sequence: {test_sequence}")
    print(f"Sequence length: {len(test_sequence)}")
    
    composition = analyze_amino_acid_composition(test_sequence)
    
    print(f"Total amino acids: {composition['total_amino_acids']}")
    print("Amino acid counts:")
    for aa, count in composition['counts'].items():
        if count > 0:
            print(f"  {aa}: {count} ({composition['percentages'][aa]:.2f}%)")
    
    print("Group composition:")
    for group, count in composition['groups'].items():
        print(f"  {group}: {count} ({composition['group_percentages'][group]:.2f}%)")
