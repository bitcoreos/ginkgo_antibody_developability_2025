"""
Thermal Stability Prediction Module for FLAb Framework

This module provides functionality for predicting the thermal stability of protein sequences.
"""

# Known thermostable motifs and features
THERMOSTABLE_MOTIFS = [
    'CC',      # Disulfide bonds
    'PP',      # Proline-Proline motifs
    'GG',      # Glycine-Glycine motifs
    'AA',      # Alanine-Alanine motifs
    'LL',      # Leucine-Leucine motifs
    'VV',      # Valine-Valine motifs
    'II',      # Isoleucine-Isoleucine motifs
]

# Amino acids that contribute to thermostability
THERMOSTABLE_AMINO_ACIDS = ['A', 'P', 'G', 'V', 'L', 'I', 'M']

# Amino acids that reduce thermostability
THERMOLABILE_AMINO_ACIDS = ['D', 'E', 'N', 'Q', 'H', 'K', 'R', 'S', 'T', 'Y']

def predict_thermal_stability(sequence):
    """
    Predict the thermal stability of a protein sequence based on composition and motifs.
    
    Args:
        sequence (str): Protein sequence
        
    Returns:
        dict: Thermal stability prediction including:
            - score: Normalized stability score (0-1)
            - features: Dictionary of stability-contributing features
    """
    # Convert sequence to uppercase
    sequence = sequence.upper()
    
    # Initialize feature counters
    features = {
        'disulfide_bonds': 0,
        'thermostable_motifs': 0,
        'thermostable_aa': 0,
        'thermolabile_aa': 0,
        'sequence_length': len(sequence)
    }
    
    # Count disulfide bonds (cysteine pairs)
    cysteine_count = sequence.count('C')
    features['disulfide_bonds'] = cysteine_count // 2
    
    # Count thermostable motifs
    for motif in THERMOSTABLE_MOTIFS:
        features['thermostable_motifs'] += sequence.count(motif)
    
    # Count thermostable amino acids
    for aa in THERMOSTABLE_AMINO_ACIDS:
        features['thermostable_aa'] += sequence.count(aa)
    
    # Count thermolabile amino acids
    for aa in THERMOLABILE_AMINO_ACIDS:
        features['thermolabile_aa'] += sequence.count(aa)
    
    # Calculate stability score based on features
    # Start with a neutral score
    score = 0.5
    
    # Adjust for disulfide bonds (positive contribution)
    score += features['disulfide_bonds'] * 0.05
    
    # Adjust for thermostable motifs (positive contribution)
    score += features['thermostable_motifs'] * 0.02
    
    # Adjust for thermostable amino acid composition (positive contribution)
    if len(sequence) > 0:
        thermostable_ratio = features['thermostable_aa'] / len(sequence)
        score += thermostable_ratio * 0.3
    
    # Adjust for thermolabile amino acid composition (negative contribution)
    if len(sequence) > 0:
        thermolabile_ratio = features['thermolabile_aa'] / len(sequence)
        score -= thermolabile_ratio * 0.3
    
    # Clamp score between 0 and 1
    score = max(0.0, min(1.0, score))
    
    return {
        'score': score,
        'features': features
    }

def analyze_thermal_stability_features(sequence):
    """
    Analyze specific features that contribute to thermal stability.
    
    Args:
        sequence (str): Protein sequence
        
    Returns:
        dict: Detailed analysis of thermal stability features
    """
    # Convert sequence to uppercase
    sequence = sequence.upper()
    
    # Analyze specific features
    analysis = {
        'disulfide_bonds': {
            'count': sequence.count('C') // 2,
            'description': 'Cysteine pairs that can form disulfide bonds'
        },
        'proline_content': {
            'count': sequence.count('P'),
            'percentage': (sequence.count('P') / len(sequence)) * 100 if len(sequence) > 0 else 0,
            'description': 'Proline residues that increase rigidity'
        },
        'glycine_content': {
            'count': sequence.count('G'),
            'percentage': (sequence.count('G') / len(sequence)) * 100 if len(sequence) > 0 else 0,
            'description': 'Glycine residues that increase flexibility'
        },
        'hydrophobicity': {
            'score': calculate_hydrophobicity_fraction(sequence),
            'description': 'Fraction of hydrophobic residues'
        }
    }
    
    return analysis

def calculate_hydrophobicity_fraction(sequence):
    """
    Calculate the fraction of hydrophobic residues in a sequence.
    
    Args:
        sequence (str): Protein sequence
        
    Returns:
        float: Fraction of hydrophobic residues
    """
    # Convert sequence to uppercase
    sequence = sequence.upper()
    
    # Define hydrophobic amino acids
    hydrophobic_aa = ['A', 'F', 'G', 'I', 'L', 'M', 'P', 'V', 'W', 'Y']
    
    # Count hydrophobic residues
    hydrophobic_count = 0
    for aa in sequence:
        if aa in hydrophobic_aa:
            hydrophobic_count += 1
    
    # Calculate fraction
    fraction = hydrophobic_count / len(sequence) if len(sequence) > 0 else 0
    
    return fraction

if __name__ == "__main__":
    # Test the thermal stability prediction
    test_sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDFWSGYWGQGTLVTVSS"
    
    print("Testing thermal stability prediction:")
    print(f"Test sequence: {test_sequence}")
    print(f"Sequence length: {len(test_sequence)}")
    
    # Predict thermal stability
    stability = predict_thermal_stability(test_sequence)
    print(f"\nThermal stability score: {stability['score']:.4f}")
    print("Stability features:")
    for feature, value in stability['features'].items():
        print(f"  {feature}: {value}")
    
    # Analyze specific features
    feature_analysis = analyze_thermal_stability_features(test_sequence)
    print("\nDetailed feature analysis:")
    for feature, data in feature_analysis.items():
        if feature in ['disulfide_bonds']:
            print(f"  {feature}: {data['count']} - {data['description']}")
        elif feature in ['proline_content', 'glycine_content']:
            print(f"  {feature}: {data['count']} ({data['percentage']:.2f}%) - {data['description']}")
        else:
            print(f"  {feature}: {data['score']:.4f} - {data['description']}")
