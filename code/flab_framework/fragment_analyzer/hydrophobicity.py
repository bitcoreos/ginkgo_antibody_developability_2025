"""
Hydrophobicity Calculation Module for FLAb Framework

This module provides functionality for calculating hydrophobicity of protein sequences
using the Kyte-Doolittle hydrophobicity scale.
"""

# Kyte-Doolittle hydrophobicity scale (at pH 7.0)
# Values from Kyte, J. and Doolittle, R.F. (1982)
HYDROPHOBICITY_SCALE = {
    'A': 1.8,   # Alanine
    'R': -4.5,  # Arginine
    'N': -3.5,  # Asparagine
    'D': -3.5,  # Aspartic acid
    'C': 2.5,   # Cysteine
    'Q': -3.5,  # Glutamine
    'E': -3.5,  # Glutamic acid
    'G': -0.4,  # Glycine
    'H': -3.2,  # Histidine
    'I': 4.5,   # Isoleucine
    'L': 3.8,   # Leucine
    'K': -3.9,  # Lysine
    'M': 1.9,   # Methionine
    'F': 2.8,   # Phenylalanine
    'P': -1.6,  # Proline
    'S': -0.8,  # Serine
    'T': -0.7,  # Threonine
    'W': -0.9,  # Tryptophan
    'Y': -1.3,  # Tyrosine
    'V': 4.2    # Valine
}

def calculate_hydrophobicity(sequence):
    """
    Calculate the average hydrophobicity of a protein sequence using the Kyte-Doolittle scale.
    
    Args:
        sequence (str): Protein sequence
        
    Returns:
        float: Average hydrophobicity score
            - Positive values indicate hydrophobicity
            - Negative values indicate hydrophilicity
    """
    # Convert sequence to uppercase
    sequence = sequence.upper()
    
    # Calculate total hydrophobicity
    total_hydrophobicity = 0
    valid_residues = 0
    
    # Analyze each residue in the sequence
    for aa in sequence:
        if aa in HYDROPHOBICITY_SCALE:
            total_hydrophobicity += HYDROPHOBICITY_SCALE[aa]
            valid_residues += 1
    
    # Calculate average hydrophobicity
    average_hydrophobicity = total_hydrophobicity / valid_residues if valid_residues > 0 else 0
    
    return average_hydrophobicity

def calculate_hydrophobicity_profile(sequence, window_size=5):
    """
    Calculate a sliding window hydrophobicity profile for a protein sequence.
    
    Args:
        sequence (str): Protein sequence
        window_size (int): Size of the sliding window (default: 5)
        
    Returns:
        list: Hydrophobicity scores for each window position
    """
    # Convert sequence to uppercase
    sequence = sequence.upper()
    
    # Calculate hydrophobicity profile
    profile = []
    
    # For each window position
    for i in range(len(sequence) - window_size + 1):
        # Extract window
        window = sequence[i:i+window_size]
        
        # Calculate average hydrophobicity for window
        window_hydrophobicity = 0
        valid_residues = 0
        
        for aa in window:
            if aa in HYDROPHOBICITY_SCALE:
                window_hydrophobicity += HYDROPHOBICITY_SCALE[aa]
                valid_residues += 1
        
        # Calculate average for window
        average_window_hydrophobicity = window_hydrophobicity / valid_residues if valid_residues > 0 else 0
        profile.append(average_window_hydrophobicity)
    
    return profile

def find_hydrophobic_regions(sequence, threshold=1.0, min_length=4):
    """
    Identify hydrophobic regions in a protein sequence.
    
    Args:
        sequence (str): Protein sequence
        threshold (float): Minimum hydrophobicity score to consider hydrophobic (default: 1.0)
        min_length (int): Minimum length of hydrophobic region (default: 4)
        
    Returns:
        list: List of hydrophobic regions with start position, end position, and average hydrophobicity
    """
    # Convert sequence to uppercase
    sequence = sequence.upper()
    
    # Calculate hydrophobicity profile with window size of 1 (individual residues)
    profile = []
    for aa in sequence:
        if aa in HYDROPHOBICITY_SCALE:
            profile.append(HYDROPHOBICITY_SCALE[aa])
        else:
            profile.append(0)  # Unknown amino acid
    
    # Identify hydrophobic regions
    regions = []
    in_region = False
    region_start = 0
    region_scores = []
    
    for i, score in enumerate(profile):
        if score >= threshold and not in_region:
            # Start of a new hydrophobic region
            in_region = True
            region_start = i
            region_scores = [score]
        elif score >= threshold and in_region:
            # Continue current hydrophobic region
            region_scores.append(score)
        elif score < threshold and in_region:
            # End of current hydrophobic region
            if len(region_scores) >= min_length:
                # Region meets minimum length requirement
                avg_score = sum(region_scores) / len(region_scores)
                regions.append({
                    'start': region_start,
                    'end': i-1,
                    'length': len(region_scores),
                    'average_hydrophobicity': avg_score
                })
            in_region = False
            region_scores = []
    
    # Check if we're still in a region at the end of the sequence
    if in_region and len(region_scores) >= min_length:
        avg_score = sum(region_scores) / len(region_scores)
        regions.append({
            'start': region_start,
            'end': len(sequence)-1,
            'length': len(region_scores),
            'average_hydrophobicity': avg_score
        })
    
    return regions

if __name__ == "__main__":
    # Test the hydrophobicity calculation
    test_sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDFWSGYWGQGTLVTVSS"
    
    print("Testing hydrophobicity calculation:")
    print(f"Test sequence: {test_sequence}")
    print(f"Sequence length: {len(test_sequence)}")
    
    # Calculate average hydrophobicity
    avg_hydrophobicity = calculate_hydrophobicity(test_sequence)
    print(f"Average hydrophobicity: {avg_hydrophobicity:.4f}")
    
    # Calculate hydrophobicity profile
    profile = calculate_hydrophobicity_profile(test_sequence, window_size=5)
    print(f"Hydrophobicity profile (5-residue window): {len(profile)} values")
    print(f"First 10 profile values: {[f'{x:.2f}' for x in profile[:10]]}")
    
    # Find hydrophobic regions
    regions = find_hydrophobic_regions(test_sequence, threshold=1.0, min_length=4)
    print(f"Hydrophobic regions (threshold=1.0, min_length=4): {len(regions)} found")
    for i, region in enumerate(regions[:3]):  # Show first 3 regions
        print(f"  Region {i+1}: Position {region['start']}-{region['end']} (length {region['length']}, avg hydrophobicity {region['average_hydrophobicity']:.2f})")
