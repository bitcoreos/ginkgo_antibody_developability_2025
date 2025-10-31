
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

# H-bond donors and acceptors
h_bond_donors = set(['R', 'K', 'H', 'N', 'Q', 'S', 'T', 'Y', 'W'])
h_bond_acceptors = set(['D', 'E', 'H', 'N', 'Q', 'S', 'T', 'Y', 'W'])

def calculate_ch1_cl_stability_score(row):
    heavy_aho = row['heavy_aligned_aho']
    light_aho = row['light_aligned_aho']

    # Extract CH1 domain from heavy chain
    ch1_start = 100
    ch1_end = 180
    ch1_seq = heavy_aho[ch1_start:ch1_end].replace('-', '')

    # Extract CL domain from light chain
    cl_start = 90
    cl_end = 180
    cl_seq = light_aho[cl_start:cl_end].replace('-', '')

    # Calculate hydrophobicity score
    if len(ch1_seq) == 0 or len(cl_seq) == 0:
        hydrophobicity_score = 0
    else:
        ch1_hydrophobicity = np.mean([kyte_doolittle.get(aa, 0) for aa in ch1_seq])
        cl_hydrophobicity = np.mean([kyte_doolittle.get(aa, 0) for aa in cl_seq])
        hydrophobicity_score = (ch1_hydrophobicity + cl_hydrophobicity) / 2

    # Calculate H-bond potential
    interface_seq = ch1_seq + cl_seq
    if len(interface_seq) == 0:
        h_bond_score = 0
    else:
        h_bond_density = sum(1 for aa in interface_seq if aa in h_bond_donors or aa in h_bond_acceptors) / len(interface_seq)
        h_bond_score = h_bond_density * 10

    # Final score: 0.7 × Hydrophobicity + 0.3 × H-bond count
    final_score = 0.7 * hydrophobicity_score + 0.3 * h_bond_score

    return pd.Series({
        'ch1_cl_hydrophobicity': hydrophobicity_score,
        'ch1_cl_hbond_density': h_bond_score,
        'ch1_cl_stability_score': final_score
    })

# Apply the function
df_scores = df_sequences.apply(calculate_ch1_cl_stability_score, axis=1)

df_results = pd.concat([df_sequences, df_scores], axis=1)

df_results.to_csv('/a0/bitcore/workspace/data/features/ch1_cl_stability_features.csv', index=False)
