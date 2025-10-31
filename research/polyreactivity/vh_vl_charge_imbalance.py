"""
VH/VL Charge Imbalance Implementation

This module implements advanced charge analysis for antibody sequences,
focusing on the charge imbalance between heavy (VH) and light (VL) chains.
"""

import numpy as np
from typing import Dict, List, Tuple, Union


def calculate_net_charge(sequence: str) -> float:
    """
    Calculate the net charge of a protein sequence at neutral pH (pH 7.4).
    
    Args:
        sequence (str): Protein sequence
        
    Returns:
        float: Net charge
    """
    # Amino acid charges at pH 7.4
    # Positive charges
    positive_aa = {'K': 1, 'R': 1, 'H': 0.1}  # Lysine, Arginine, Histidine (partial)
    # Negative charges
    negative_aa = {'D': -1, 'E': -1}  # Aspartic acid, Glutamic acid
    
    net_charge = 0.0
    
    for aa in sequence:
        if aa in positive_aa:
            net_charge += positive_aa[aa]
        elif aa in negative_aa:
            net_charge += negative_aa[aa]
    
    return net_charge


def calculate_charge_distribution(sequence: str) -> Dict[str, float]:
    """
    Calculate detailed charge distribution in a protein sequence.
    
    Args:
        sequence (str): Protein sequence
        
    Returns:
        Dict[str, float]: Charge distribution metrics
    """
    # Count charged amino acids
    positive_counts = {'K': 0, 'R': 0, 'H': 0}
    negative_counts = {'D': 0, 'E': 0}
    
    for aa in sequence:
        if aa in positive_counts:
            positive_counts[aa] += 1
        elif aa in negative_counts:
            negative_counts[aa] += 1
    
    # Calculate distribution metrics
    total_positive = sum(positive_counts.values())
    total_negative = sum(negative_counts.values())
    total_charged = total_positive + total_negative
    
    # Avoid division by zero
    if total_charged == 0:
        positive_ratio = 0.0
        negative_ratio = 0.0
    else:
        positive_ratio = total_positive / total_charged
        negative_ratio = total_negative / total_charged
    
    return {
        'positive_counts': positive_counts,
        'negative_counts': negative_counts,
        'total_positive': total_positive,
        'total_negative': total_negative,
        'positive_ratio': positive_ratio,
        'negative_ratio': negative_ratio,
        'charge_balance': abs(total_positive - total_negative)
    }


def calculate_vh_vl_charge_imbalance(vh_sequence: str, vl_sequence: str) -> Dict[str, Union[float, Dict]]:
    """
    Calculate VH/VL charge imbalance and related features.
    
    Args:
        vh_sequence (str): Heavy chain variable domain sequence
        vl_sequence (str): Light chain variable domain sequence
        
    Returns:
        Dict: Dictionary with charge imbalance features
    """
    # Calculate net charges
    vh_net_charge = calculate_net_charge(vh_sequence)
    vl_net_charge = calculate_net_charge(vl_sequence)
    
    # Calculate charge imbalance
    charge_imbalance = abs(vh_net_charge - vl_net_charge)
    
    # Normalize by sequence length
    vh_length = len(vh_sequence)
    vl_length = len(vl_sequence)
    
    # Avoid division by zero
    if vh_length > 0:
        vh_normalized_charge = vh_net_charge / vh_length
    else:
        vh_normalized_charge = 0.0
        
    if vl_length > 0:
        vl_normalized_charge = vl_net_charge / vl_length
    else:
        vl_normalized_charge = 0.0
    
    normalized_imbalance = abs(vh_normalized_charge - vl_normalized_charge)
    
    # Calculate charge distributions
    vh_charge_dist = calculate_charge_distribution(vh_sequence)
    vl_charge_dist = calculate_charge_distribution(vl_sequence)
    
    # Additional metrics
    total_charge_imbalance = abs(vh_charge_dist['charge_balance'] - vl_charge_dist['charge_balance'])
    
    return {
        'vh_net_charge': vh_net_charge,
        'vl_net_charge': vl_net_charge,
        'charge_imbalance': charge_imbalance,
        'vh_normalized_charge': vh_normalized_charge,
        'vl_normalized_charge': vl_normalized_charge,
        'normalized_imbalance': normalized_imbalance,
        'vh_charge_distribution': vh_charge_dist,
        'vl_charge_distribution': vl_charge_dist,
        'total_charge_imbalance': total_charge_imbalance,
        'vh_length': vh_length,
        'vl_length': vl_length
    }


def analyze_charge_clustering(sequence: str, window_size: int = 5) -> Dict[str, float]:
    """
    Analyze basic residue clustering patterns in a sequence.
    
    Args:
        sequence (str): Protein sequence
        window_size (int): Size of sliding window for clustering analysis
        
    Returns:
        Dict[str, float]: Clustering metrics
    """
    # Define charged amino acids
    positive_aa = {'K', 'R', 'H'}
    negative_aa = {'D', 'E'}
    
    # Count charged residues in sliding windows
    positive_clusters = []
    negative_clusters = []
    
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i+window_size]
        pos_count = sum(1 for aa in window if aa in positive_aa)
        neg_count = sum(1 for aa in window if aa in negative_aa)
        
        if pos_count > 0:
            positive_clusters.append(pos_count)
        if neg_count > 0:
            negative_clusters.append(neg_count)
    
    # Calculate clustering metrics
    avg_positive_cluster = np.mean(positive_clusters) if positive_clusters else 0.0
    avg_negative_cluster = np.mean(negative_clusters) if negative_clusters else 0.0
    
    max_positive_cluster = max(positive_clusters) if positive_clusters else 0
    max_negative_cluster = max(negative_clusters) if negative_clusters else 0
    
    return {
        'avg_positive_cluster': avg_positive_cluster,
        'avg_negative_cluster': avg_negative_cluster,
        'max_positive_cluster': max_positive_cluster,
        'max_negative_cluster': max_negative_cluster,
        'total_positive_clusters': len(positive_clusters),
        'total_negative_clusters': len(negative_clusters)
    }


def analyze_hydrophobic_patches(sequence: str, window_size: int = 5) -> Dict[str, float]:
    """
    Analyze hydrophobic patches in a sequence for surface binding prediction.
    
    Args:
        sequence (str): Protein sequence
        window_size (int): Size of sliding window for patch analysis
        
    Returns:
        Dict[str, float]: Hydrophobic patch metrics
    """
    # Define hydrophobic amino acids
    hydrophobic_aa = {'A', 'F', 'I', 'L', 'M', 'P', 'V', 'W', 'Y'}
    
    # Count hydrophobic residues in sliding windows
    hydrophobic_patches = []
    
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i+window_size]
        hydro_count = sum(1 for aa in window if aa in hydrophobic_aa)
        
        if hydro_count >= window_size // 2:  # At least half hydrophobic
            hydrophobic_patches.append(hydro_count)
    
    # Calculate patch metrics
    avg_patch_size = np.mean(hydrophobic_patches) if hydrophobic_patches else 0.0
    max_patch_size = max(hydrophobic_patches) if hydrophobic_patches else 0
    
    return {
        'avg_patch_size': avg_patch_size,
        'max_patch_size': max_patch_size,
        'total_patches': len(hydrophobic_patches),
        'patch_density': len(hydrophobic_patches) / len(sequence) if sequence else 0.0
    }


def main():
    """
    Example usage of the VH/VL charge imbalance implementation.
    """
    # Example VH and VL sequences (simplified)
    vh_sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    vl_sequence = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLAWYQQKPGKAPKLLIYDASNLRTGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSNHWPTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
    
    # Calculate VH/VL charge imbalance
    features = calculate_vh_vl_charge_imbalance(vh_sequence, vl_sequence)
    
    print("VH/VL Charge Imbalance Features:")
    print(f"VH Net Charge: {features['vh_net_charge']:.2f}")
    print(f"VL Net Charge: {features['vl_net_charge']:.2f}")
    print(f"Charge Imbalance: {features['charge_imbalance']:.2f}")
    print(f"Normalized Imbalance: {features['normalized_imbalance']:.4f}")
    
    # Analyze charge clustering
    vh_clustering = analyze_charge_clustering(vh_sequence)
    vl_clustering = analyze_charge_clustering(vl_sequence)
    
    print("\nCharge Clustering Analysis:")
    print(f"VH Avg Positive Cluster: {vh_clustering['avg_positive_cluster']:.2f}")
    print(f"VL Avg Negative Cluster: {vl_clustering['avg_negative_cluster']:.2f}")
    
    # Analyze hydrophobic patches
    vh_patches = analyze_hydrophobic_patches(vh_sequence)
    vl_patches = analyze_hydrophobic_patches(vl_sequence)
    
    print("\nHydrophobic Patch Analysis:")
    print(f"VH Max Patch Size: {vh_patches['max_patch_size']}")
    print(f"VL Patch Density: {vl_patches['patch_density']:.4f}")


if __name__ == "__main__":
    main()
