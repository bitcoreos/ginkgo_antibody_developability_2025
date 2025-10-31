"""
Aggregation Propensity Prediction Module for FLAb Framework

This module provides functionality for predicting the aggregation propensity of protein sequences.
"""

# Known aggregation-prone motifs
AGGREGATION_MOTIFS = [
    'FFFF',    # Phenylalanine-rich regions
    'WWWW',    # Tryptophan-rich regions
    'YYYY',    # Tyrosine-rich regions
    'AIVM',    # Hydrophobic motif
    'LFILV',   # Branched hydrophobic motif
    'VQIIL',   # Amyloid-like motif
    'NFGQ',    # Aggregation-prone motif
    'NCPPP',   # Aggregation-prone motif
]

# Amino acids that contribute to aggregation propensity
AGGREGATION_PRONE_AMINO_ACIDS = ['F', 'W', 'Y', 'I', 'L', 'V', 'M', 'A']

# Charged amino acids that can reduce aggregation
CHARGED_AMINO_ACIDS = ['R', 'H', 'K', 'D', 'E']

def predict_aggregation_propensity(sequence):
    """
    Predict the aggregation propensity of a protein sequence based on composition and motifs.
    
    Args:
        sequence (str): Protein sequence
        
    Returns:
        dict: Aggregation propensity prediction including:
            - score: Normalized aggregation propensity score (0-1)
            - features: Dictionary of aggregation-contributing features
    """
    # Convert sequence to uppercase
    sequence = sequence.upper()
    
    # Initialize feature counters
    features = {
        'aggregation_motifs': 0,
        'aggregation_prone_aa': 0,
        'charged_aa': 0,
        'sequence_length': len(sequence)
    }
    
    # Count aggregation-prone motifs
    for motif in AGGREGATION_MOTIFS:
        features['aggregation_motifs'] += sequence.count(motif)
    
    # Count aggregation-prone amino acids
    for aa in AGGREGATION_PRONE_AMINO_ACIDS:
        features['aggregation_prone_aa'] += sequence.count(aa)
    
    # Count charged amino acids
    for aa in CHARGED_AMINO_ACIDS:
        features['charged_aa'] += sequence.count(aa)
    
    # Calculate aggregation propensity score based on features
    # Start with a neutral score
    score = 0.5
    
    # Adjust for aggregation-prone motifs (positive contribution)
    score += features['aggregation_motifs'] * 0.1
    
    # Adjust for aggregation-prone amino acid composition (positive contribution)
    if len(sequence) > 0:
        aggregation_prone_ratio = features['aggregation_prone_aa'] / len(sequence)
        score += aggregation_prone_ratio * 0.3
    
    # Adjust for charged amino acid composition (negative contribution)
    if len(sequence) > 0:
        charged_ratio = features['charged_aa'] / len(sequence)
        score -= charged_ratio * 0.3
    
    # Clamp score between 0 and 1
    score = max(0.0, min(1.0, score))
    
    return {
        'score': score,
        'features': features
    }

def analyze_aggregation_features(sequence):
    """
    Analyze specific features that contribute to aggregation propensity.
    
    Args:
        sequence (str): Protein sequence
        
    Returns:
        dict: Detailed analysis of aggregation propensity features
    """
    # Convert sequence to uppercase
    sequence = sequence.upper()
    
    # Analyze specific features
    analysis = {
        'aromatic_content': {
            'count': sequence.count('F') + sequence.count('W') + sequence.count('Y'),
            'percentage': ((sequence.count('F') + sequence.count('W') + sequence.count('Y')) / len(sequence)) * 100 if len(sequence) > 0 else 0,
            'description': 'Aromatic residues that can participate in π-π stacking'
        },
        'hydrophobic_content': {
            'count': sequence.count('A') + sequence.count('I') + sequence.count('V') + sequence.count('L') + sequence.count('M'),
            'percentage': ((sequence.count('A') + sequence.count('I') + sequence.count('V') + sequence.count('L') + sequence.count('M')) / len(sequence)) * 100 if len(sequence) > 0 else 0,
            'description': 'Hydrophobic residues that can drive aggregation'
        },
        'charged_content': {
            'count': sequence.count('R') + sequence.count('H') + sequence.count('K') + sequence.count('D') + sequence.count('E'),
            'percentage': ((sequence.count('R') + sequence.count('H') + sequence.count('K') + sequence.count('D') + sequence.count('E')) / len(sequence)) * 100 if len(sequence) > 0 else 0,
            'description': 'Charged residues that can reduce aggregation'
        }
    }
    
    return analysis

def find_aggregation_prone_regions(sequence, window_size=6):
    """
    Identify aggregation-prone regions in a protein sequence.
    
    Args:
        sequence (str): Protein sequence
        window_size (int): Size of the sliding window (default: 6)
        
    Returns:
        list: List of aggregation-prone regions with start position, end position, and score
    """
    # Convert sequence to uppercase
    sequence = sequence.upper()
    
    # Define aggregation-prone amino acids
    aggregation_prone = ['F', 'W', 'Y', 'I', 'L', 'V', 'M', 'A']
    
    # Identify aggregation-prone regions
    regions = []
    
    # For each window position
    for i in range(len(sequence) - window_size + 1):
        # Extract window
        window = sequence[i:i+window_size]
        
        # Count aggregation-prone residues in window
        agg_prone_count = 0
        for aa in window:
            if aa in aggregation_prone:
                agg_prone_count += 1
        
        # Calculate aggregation propensity for window
        agg_propensity = agg_prone_count / window_size if window_size > 0 else 0
        
        # If above threshold, consider it an aggregation-prone region
        if agg_propensity > 0.66:  # More than 2/3 of residues are aggregation-prone
            regions.append({
                'start': i,
                'end': i+window_size-1,
                'length': window_size,
                'aggregation_propensity': agg_propensity,
                'sequence': window
            })
    
    return regions

if __name__ == "__main__":
    # Test the aggregation propensity prediction
    test_sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDFWSGYWGQGTLVTVSS"
    
    print("Testing aggregation propensity prediction:")
    print(f"Test sequence: {test_sequence}")
    print(f"Sequence length: {len(test_sequence)}")
    
    # Predict aggregation propensity
    aggregation = predict_aggregation_propensity(test_sequence)
    print(f"\nAggregation propensity score: {aggregation['score']:.4f}")
    print("Aggregation features:")
    for feature, value in aggregation['features'].items():
        print(f"  {feature}: {value}")
    
    # Analyze specific features
    feature_analysis = analyze_aggregation_features(test_sequence)
    print("\nDetailed feature analysis:")
    for feature, data in feature_analysis.items():
        print(f"  {feature}: {data['count']} ({data['percentage']:.2f}%) - {data['description']}")
    
    # Find aggregation-prone regions
    agg_regions = find_aggregation_prone_regions(test_sequence, window_size=6)
    print(f"\nAggregation-prone regions (window=6, threshold=0.66): {len(agg_regions)} found")
    for i, region in enumerate(agg_regions[:3]):  # Show first 3 regions
        print(f"  Region {i+1}: Position {region['start']}-{region['end']} '{region['sequence']}' (propensity: {region['aggregation_propensity']:.2f})")
