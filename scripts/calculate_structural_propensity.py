
import pandas as pd
import numpy as np

# Load the sequences data
df_sequences = pd.read_csv('/a0/bitcore/workspace/data/sequences/GDPa1_v1.2_sequences.csv')

# Kyte-Doolittle hydrophobicity scale
kyte_doolittle = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8,
    'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6,
    'H': -3.2, 'E': -3.5, 'N': -3.5, 'Q': -3.5, 'D': -3.5, 'K': -3.9, 'R': -4.5
}

def extract_cdr_regions(row):
    heavy_aho = row['heavy_aligned_aho']

    # Extract CDR-H1 (AHO positions 26-32) from heavy chain
    cdr_h1_start = 20
    cdr_h1_end = 35
    cdr_h1_seq = heavy_aho[cdr_h1_start:cdr_h1_end].replace('-', '')

    # Extract CDR-H2 (AHO positions 52-56) from heavy chain
    cdr_h2_start = 50
    cdr_h2_end = 65
    cdr_h2_seq = heavy_aho[cdr_h2_start:cdr_h2_end].replace('-', '')

    # Extract CDR-H3 (AHO positions 95-102) from heavy chain
    cdr_h3_start = 90
    cdr_h3_end = 110
    cdr_h3_seq = heavy_aho[cdr_h3_start:cdr_h3_end].replace('-', '')

    return pd.Series({
        'cdr_h1_seq': cdr_h1_seq,
        'cdr_h2_seq': cdr_h2_seq,
        'cdr_h3_seq': cdr_h3_seq
    })

# Apply the function
df_cdr = df_sequences.apply(extract_cdr_regions, axis=1)

df_with_cdr = pd.concat([df_sequences, df_cdr], axis=1)

def calculate_structural_propensity(row):
    cdr_h1_seq = row['cdr_h1_seq']
    cdr_h2_seq = row['cdr_h2_seq']
    cdr_h3_seq = row['cdr_h3_seq']

    # Combine all CDR sequences
    all_cdr_seq = cdr_h1_seq + cdr_h2_seq + cdr_h3_seq

    # 1. Electrostatic potential: sum of positive charges (Arg, Lys) in CDR-H3
    electrostatic_potential = cdr_h3_seq.count('R') + cdr_h3_seq.count('K')

    # 2. Hydrophobicity moment: average Kyte-Doolittle score of all CDR regions
    if len(all_cdr_seq) == 0:
        hydrophobicity_moment = 0
    else:
        hydrophobicity_moment = np.mean([kyte_doolittle.get(aa, 0) for aa in all_cdr_seq])

    # 3. Aromatic cluster density: count of aromatic residues in CDR-H3
    aromatic_residues = ['F', 'Y', 'W']
    aromatic_count = sum(1 for aa in cdr_h3_seq if aa in aromatic_residues)
    if len(cdr_h3_seq) == 0:
        aromatic_density = 0
    else:
        aromatic_density = aromatic_count / len(cdr_h3_seq)

    return pd.Series({
        'electrostatic_potential': electrostatic_potential,
        'hydrophobicity_moment': hydrophobicity_moment,
        'aromatic_cluster_density': aromatic_density
    })

# Apply the function
df_propensity = df_with_cdr.apply(calculate_structural_propensity, axis=1)

df_results = pd.concat([df_with_cdr, df_propensity], axis=1)

df_results.to_csv('/a0/bitcore/workspace/data/features/structural_propensity_features.csv', index=False)
