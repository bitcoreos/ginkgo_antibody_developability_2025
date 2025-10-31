
import pandas as pd
import numpy as np
import re

# Load the heldout set sequences
sequences_df = pd.read_csv('/a0/bitcore/workspace/data/sequences/heldout_set_sequences.csv')

# Define amino acid properties
aa_properties = {
    'A': {'hydrophobicity': 1.8, 'charge': 0, 'aromatic': 0},
    'R': {'hydrophobicity': -4.5, 'charge': 1, 'aromatic': 0},
    'N': {'hydrophobicity': -3.5, 'charge': 0, 'aromatic': 0},
    'D': {'hydrophobicity': -3.5, 'charge': -1, 'aromatic': 0},
    'C': {'hydrophobicity': 2.5, 'charge': 0, 'aromatic': 0},
    'Q': {'hydrophobicity': -3.5, 'charge': 0, 'aromatic': 0},
    'E': {'hydrophobicity': -3.5, 'charge': -1, 'aromatic': 0},
    'G': {'hydrophobicity': -0.4, 'charge': 0, 'aromatic': 0},
    'H': {'hydrophobicity': -3.2, 'charge': 0.5, 'aromatic': 1},
    'I': {'hydrophobicity': 4.5, 'charge': 0, 'aromatic': 0},
    'L': {'hydrophobicity': 3.8, 'charge': 0, 'aromatic': 0},
    'K': {'hydrophobicity': -3.9, 'charge': 1, 'aromatic': 0},
    'M': {'hydrophobicity': 1.9, 'charge': 0, 'aromatic': 0},
    'F': {'hydrophobicity': 2.8, 'charge': 0, 'aromatic': 1},
    'P': {'hydrophobicity': -1.6, 'charge': 0, 'aromatic': 0},
    'S': {'hydrophobicity': -0.8, 'charge': 0, 'aromatic': 0},
    'T': {'hydrophobicity': -0.7, 'charge': 0, 'aromatic': 0},
    'W': {'hydrophobicity': -0.9, 'charge': 0, 'aromatic': 1},
    'Y': {'hydrophobicity': -1.3, 'charge': 0, 'aromatic': 1},
    'V': {'hydrophobicity': 4.2, 'charge': 0, 'aromatic': 0}
}

# Function to extract CDR regions using regex patterns
def extract_cdr_regions(sequence):
    # This is a simplified approach - in practice, you'd use a tool like ANARCI
    # For now, we'll use regex patterns to find CDR regions based on common motifs
    # This is a placeholder implementation and may not be accurate for all sequences

    # Find CDR-H1: typically between positions 26-35 in VH
    cdr_h1_match = re.search(r'([GSTNA]..[GSTNA]...[GSTNA]...[GSTNA])', sequence)
    cdr_h1 = cdr_h1_match.group(1) if cdr_h1_match else ''

    # Find CDR-H2: typically between positions 50-65 in VH
    cdr_h2_match = re.search(r'([WYF][^P]{5,7}[GW][^P]{2,4}[FYW])', sequence)
    cdr_h2 = cdr_h2_match.group(1) if cdr_h2_match else ''

    # Find CDR-H3: typically between positions 95-102 in VH, most variable
    # Look for conserved residues at start (e.g., W) and end (e.g., WG, WGX)
    cdr_h3_match = re.search(r'W[^P]{5,25}[WG][^P]{0,3}[FYW]', sequence)
    cdr_h3 = cdr_h3_match.group(0) if cdr_h3_match else ''

    return cdr_h1, cdr_h2, cdr_h3

# Function to calculate structural propensity scores
def calculate_structural_propensity(cdr_h1, cdr_h2, cdr_h3):
    # Combine all CDR sequences
    all_cdrs = cdr_h1 + cdr_h2 + cdr_h3

    if not all_cdrs:
        return 0, 0, 0  # Return zeros if no CDRs found

    # Calculate electrostatic potential (sum of charges)
    electrostatic_potential = sum(aa_properties.get(aa, {'charge': 0})['charge'] for aa in all_cdrs)

    # Calculate hydrophobicity moment (average hydrophobicity)
    if len(all_cdrs) > 0:
        hydrophobicity_sum = sum(aa_properties.get(aa, {'hydrophobicity': 0})['hydrophobicity'] for aa in all_cdrs)
        hydrophobicity_moment = hydrophobicity_sum / len(all_cdrs)
    else:
        hydrophobicity_moment = 0

    # Calculate aromatic cluster density (count of aromatic residues)
    aromatic_count = sum(aa_properties.get(aa, {'aromatic': 0})['aromatic'] for aa in all_cdrs)
    aromatic_cluster_density = aromatic_count / len(all_cdrs) if len(all_cdrs) > 0 else 0

    return electrostatic_potential, hydrophobicity_moment, aromatic_cluster_density

# Process each sequence
results = []
for idx, row in sequences_df.iterrows():
    sequence_id = row['antibody_name']
    sequence = row['vh_protein_sequence']

    # Extract CDR regions
    cdr_h1, cdr_h2, cdr_h3 = extract_cdr_regions(sequence)

    # Calculate structural propensity scores
    ep, hm, acd = calculate_structural_propensity(cdr_h1, cdr_h2, cdr_h3)

    results.append({
        'antibody_id': sequence_id,
        'electrostatic_potential': ep,
        'hydrophobicity_moment': hm,
        'aromatic_cluster_density': acd
    })

# Create DataFrame with results
features_df = pd.DataFrame(results)

# Save to CSV
output_path = '/a0/bitcore/workspace/data/features/structural_propensity_features_heldout.csv'
features_df.to_csv(output_path, index=False)
