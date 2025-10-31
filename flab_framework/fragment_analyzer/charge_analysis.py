"""
Charge Analysis Module for FLAb Framework

This module provides functionality for analyzing charge distribution in protein sequences.
"""

# Charged amino acids and their charges at neutral pH
CHARGED_AMINO_ACIDS = {
    'R': 1,  # Arginine - positive
    'H': 1,  # Histidine - positive
    'K': 1,  # Lysine - positive
    'D': -1, # Aspartic acid - negative
    'E': -1  # Glutamic acid - negative
}

def analyze_charge_distribution(sequence):
    """
    Analyze charge distribution in a protein sequence.
    
    Args:
        sequence (str): Protein sequence
        
    Returns:
        dict: Charge distribution analysis including:
            - positive_charges: Number of positive charges
            - negative_charges: Number of negative charges
            - net_charge: Net charge of the sequence
            - charge_density: Charge density (charges per residue)
            - charged_residues: List of charged residues with positions
    """
    # Convert sequence to uppercase
    sequence = sequence.upper()
    
    # Initialize counters
    positive_charges = 0
    negative_charges = 0
    charged_residues = []
    
    # Analyze each residue in the sequence
    for i, aa in enumerate(sequence):
        if aa in CHARGED_AMINO_ACIDS:
            charge = CHARGED_AMINO_ACIDS[aa]
            if charge > 0:
                positive_charges += 1
            else:
                negative_charges += 1
            charged_residues.append({
                'position': i,
                'residue': aa,
                'charge': charge
            })
    
    # Calculate net charge
    net_charge = positive_charges - negative_charges
    
    # Calculate charge density
    charge_density = (positive_charges + negative_charges) / len(sequence) if len(sequence) > 0 else 0
    
    return {
        'positive_charges': positive_charges,
        'negative_charges': negative_charges,
        'net_charge': net_charge,
        'charge_density': charge_density,
        'charged_residues': charged_residues
    }

if __name__ == "__main__":
    # Test the charge distribution analysis
    test_sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDFWSGYWGQGTLVTVSS"
    
    print("Testing charge distribution analysis:")
    print(f"Test sequence: {test_sequence}")
    print(f"Sequence length: {len(test_sequence)}")
    
    charge_analysis = analyze_charge_distribution(test_sequence)
    
    print(f"Positive charges: {charge_analysis['positive_charges']}")
    print(f"Negative charges: {charge_analysis['negative_charges']}")
    print(f"Net charge: {charge_analysis['net_charge']}")
    print(f"Charge density: {charge_analysis['charge_density']:.4f}")
    print(f"Number of charged residues: {len(charge_analysis['charged_residues'])}")
    
    # Print first few charged residues
    print("First few charged residues:")
    for residue in charge_analysis['charged_residues'][:5]:
        print(f"  Position {residue['position']}: {residue['residue']} (charge: {residue['charge']})")
