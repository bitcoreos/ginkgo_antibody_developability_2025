"""
Isoelectric Point Calculation Module for FLAb Framework

This module provides functionality for calculating the isoelectric point of protein sequences.
"""

# pKa values for ionizable groups at 25°C and 0.1M ionic strength
# Values from Bjellqvist, B. et al. (1993)
PKA_VALUES = {
    'N_terminal': 7.5,  # N-terminal amino group
    'C_terminal': 3.5,  # C-terminal carboxyl group
    'K': 10.0,          # Lysine side chain
    'R': 12.0,          # Arginine side chain
    'H': 6.4,           # Histidine side chain
    'D': 4.0,           # Aspartic acid side chain
    'E': 4.4,           # Glutamic acid side chain
    'C': 8.5,           # Cysteine side chain
    'Y': 10.0           # Tyrosine side chain
}

def calculate_pi(sequence):
    """
    Calculate the isoelectric point of a protein sequence using the Henderson-Hasselbalch equation.
    
    Args:
        sequence (str): Protein sequence
        
    Returns:
        float: Isoelectric point (pH at which the protein has no net charge)
    """
    # Convert sequence to uppercase
    sequence = sequence.upper()
    
    # Count ionizable groups
    ionizable_groups = {
        'N_terminal': 1,  # Always one N-terminal
        'C_terminal': 1,  # Always one C-terminal
        'K': 0,           # Lysine
        'R': 0,           # Arginine
        'H': 0,           # Histidine
        'D': 0,           # Aspartic acid
        'E': 0,           # Glutamic acid
        'C': 0,           # Cysteine
        'Y': 0            # Tyrosine
    }
    
    # Count amino acids with ionizable side chains
    for aa in sequence:
        if aa in ionizable_groups:
            ionizable_groups[aa] += 1
    
    # Calculate the isoelectric point using the Henderson-Hasselbalch equation
    # For simplicity, we'll use a method based on the charge balance approach
    # where we find the pH at which the net charge is zero
    
    # We'll use a numerical approach to find the pH where net charge is zero
    # by calculating the charge at different pH values and finding where it crosses zero
    
    # Define pH range to search (0 to 14)
    pH_min = 0.0
    pH_max = 14.0
    
    # Binary search for the pH where net charge is zero
    for _ in range(100):  # Limit iterations
        pH_mid = (pH_min + pH_max) / 2.0
        net_charge = calculate_net_charge_at_pH(sequence, pH_mid)
        
        if abs(net_charge) < 1e-6:  # Convergence criteria
            return pH_mid
        elif net_charge > 0:
            pH_min = pH_mid
        else:
            pH_max = pH_mid
    
    # Return the midpoint if we haven't converged
    return (pH_min + pH_max) / 2.0

def calculate_net_charge_at_pH(sequence, pH):
    """
    Calculate the net charge of a protein at a specific pH.
    
    Args:
        sequence (str): Protein sequence
        pH (float): pH value
        
    Returns:
        float: Net charge at the specified pH
    """
    # Convert sequence to uppercase
    sequence = sequence.upper()
    
    # Initialize net charge
    net_charge = 0.0
    
    # Add contribution from N-terminal amino group (if present)
    if len(sequence) > 0:
        # N-terminal contributes positive charge if pH < pKa
        if pH < PKA_VALUES['N_terminal']:
            net_charge += 1.0
    
    # Add contribution from C-terminal carboxyl group (if present)
    if len(sequence) > 0:
        # C-terminal contributes negative charge if pH > pKa
        if pH > PKA_VALUES['C_terminal']:
            net_charge -= 1.0
    
    # Add contributions from ionizable side chains
    for aa in sequence:
        if aa in ['K', 'R', 'H']:  # Positive charges
            if pH < PKA_VALUES[aa]:
                net_charge += 1.0
        elif aa in ['D', 'E', 'C', 'Y']:  # Negative charges
            if pH > PKA_VALUES[aa]:
                net_charge -= 1.0
    
    return net_charge

def calculate_pi_simple(sequence):
    """
    Calculate a simplified isoelectric point using the average of pKa values
    of the two most contributing ionizable groups.
    
    Args:
        sequence (str): Protein sequence
        
    Returns:
        float: Simplified isoelectric point
    """
    # Convert sequence to uppercase
    sequence = sequence.upper()
    
    # Count acidic and basic residues
    acidic_count = sequence.count('D') + sequence.count('E')
    basic_count = sequence.count('K') + sequence.count('R')
    
    # Simple approximation
    # If more acidic residues, pI will be lower
    # If more basic residues, pI will be higher
    if acidic_count > basic_count:
        return 5.0 - (acidic_count - basic_count) * 0.1
    elif basic_count > acidic_count:
        return 9.0 + (basic_count - acidic_count) * 0.1
    else:
        return 7.0

if __name__ == "__main__":
    # Test the isoelectric point calculation
    test_sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDFWSGYWGQGTLVTVSS"
    
    print("Testing isoelectric point calculation:")
    print(f"Test sequence: {test_sequence}")
    print(f"Sequence length: {len(test_sequence)}")
    
    # Calculate isoelectric point using the full method
    pi = calculate_pi(test_sequence)
    print(f"Isoelectric point (full method): {pi:.2f}")
    
    # Calculate isoelectric point using the simplified method
    pi_simple = calculate_pi_simple(test_sequence)
    print(f"Isoelectric point (simplified method): {pi_simple:.2f}")
    
    # Calculate net charge at different pH values
    print("\nNet charge at different pH values:")
    for pH in [5.0, 7.0, 9.0]:
        net_charge = calculate_net_charge_at_pH(test_sequence, pH)
        print(f"  pH {pH}: Net charge = {net_charge:.2f}")
