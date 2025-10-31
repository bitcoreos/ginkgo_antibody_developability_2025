"""
Residue Clustering Pattern Analysis Module

This module implements advanced residue clustering pattern analysis for antibody variable domains.
"""

import numpy as np
from typing import Dict, List, Tuple, Union
from collections import Counter
from scipy.spatial.distance import pdist, squareform

# Amino acid property groups
CHARGED_AA = 'KRHDE'  # Lysine, Arginine, Histidine, Aspartic acid, Glutamic acid
HYDROPHOBIC_AA = 'AILMFPWV'  # Alanine, Isoleucine, Leucine, Methionine, Phenylalanine, Proline, Tryptophan, Valine
AROMATIC_AA = 'FWY'  # Phenylalanine, Tryptophan, Tyrosine


class ResidueClusteringAnalyzer:
    """
    Analyzer for residue clustering patterns in antibody variable domains.
    """
    
    def __init__(self):
        """
        Initialize the residue clustering analyzer.
        """
        pass
    
    def analyze_residue_clustering(self, sequence: str) -> Dict[str, Union[int, float, Dict]]:
        """
        Analyze residue clustering patterns in a sequence.
        
        Args:
            sequence (str): Amino acid sequence
            
        Returns:
            Dict: Residue clustering analysis results
        """
        sequence = sequence.upper()
        length = len(sequence)
        
        if length == 0:
            return {
                'sequence': sequence,
                'length': 0,
                'charged_clusters': [],
                'hydrophobic_clusters': [],
                'aromatic_clusters': [],
                'cluster_metrics': {},
                'analysis_complete': True
            }
        
        # Analyze clustering for each residue type
        charged_clusters = self._find_residue_clusters(sequence, CHARGED_AA)
        hydrophobic_clusters = self._find_residue_clusters(sequence, HYDROPHOBIC_AA)
        aromatic_clusters = self._find_residue_clusters(sequence, AROMATIC_AA)
        
        # Calculate cluster metrics
        cluster_metrics = self._calculate_cluster_metrics(
            sequence, charged_clusters, hydrophobic_clusters, aromatic_clusters
        )
        
        return {
            'sequence': sequence,
            'length': length,
            'charged_clusters': charged_clusters,
            'hydrophobic_clusters': hydrophobic_clusters,
            'aromatic_clusters': aromatic_clusters,
            'cluster_metrics': cluster_metrics,
            'analysis_complete': True
        }
    
    def _find_residue_clusters(self, sequence: str, residue_group: str) -> List[Dict]:
        """
        Find clusters of specified residue types in a sequence.
        
        Args:
            sequence (str): Amino acid sequence
            residue_group (str): Residue types to cluster
            
        Returns:
            List[Dict]: List of cluster information
        """
        clusters = []
        in_cluster = False
        cluster_start = -1
        
        for i, aa in enumerate(sequence):
            if aa in residue_group:
                if not in_cluster:
                    # Start new cluster
                    in_cluster = True
                    cluster_start = i
            else:
                if in_cluster:
                    # End current cluster
                    in_cluster = False
                    cluster_length = i - cluster_start
                    if cluster_length >= 2:  # Only consider clusters of 2 or more residues
                        clusters.append({
                            'start': cluster_start,
                            'end': i - 1,
                            'length': cluster_length,
                            'residues': sequence[cluster_start:i]
                        })
        
        # Handle case where sequence ends in a cluster
        if in_cluster:
            cluster_length = len(sequence) - cluster_start
            if cluster_length >= 2:  # Only consider clusters of 2 or more residues
                clusters.append({
                    'start': cluster_start,
                    'end': len(sequence) - 1,
                    'length': cluster_length,
                    'residues': sequence[cluster_start:]
                })
        
        return clusters
    
    def _calculate_cluster_metrics(self, sequence: str, charged_clusters: List[Dict], 
                               hydrophobic_clusters: List[Dict], aromatic_clusters: List[Dict]) -> Dict[str, Union[int, float]]:
        """
        Calculate metrics for residue clustering.
        
        Args:
            sequence (str): Amino acid sequence
            charged_clusters (List[Dict]): Charged residue clusters
            hydrophobic_clusters (List[Dict]): Hydrophobic residue clusters
            aromatic_clusters (List[Dict]): Aromatic residue clusters
            
        Returns:
            Dict: Cluster metrics
        """
        length = len(sequence)
        
        if length == 0:
            return {}
        
        # Count clusters
        charged_cluster_count = len(charged_clusters)
        hydrophobic_cluster_count = len(hydrophobic_clusters)
        aromatic_cluster_count = len(aromatic_clusters)
        
        # Calculate cluster densities
        charged_cluster_density = charged_cluster_count / length if length > 0 else 0
        hydrophobic_cluster_density = hydrophobic_cluster_count / length if length > 0 else 0
        aromatic_cluster_density = aromatic_cluster_count / length if length > 0 else 0
        
        # Calculate average cluster sizes
        avg_charged_cluster_size = np.mean([c['length'] for c in charged_clusters]) if charged_clusters else 0
        avg_hydrophobic_cluster_size = np.mean([c['length'] for c in hydrophobic_clusters]) if hydrophobic_clusters else 0
        avg_aromatic_cluster_size = np.mean([c['length'] for c in aromatic_clusters]) if aromatic_clusters else 0
        
        # Calculate maximum cluster sizes
        max_charged_cluster_size = max([c['length'] for c in charged_clusters]) if charged_clusters else 0
        max_hydrophobic_cluster_size = max([c['length'] for c in hydrophobic_clusters]) if hydrophobic_clusters else 0
        max_aromatic_cluster_size = max([c['length'] for c in aromatic_clusters]) if aromatic_clusters else 0
        
        # Calculate clustering scores (0-1, higher means more clustering)
        charged_clustering_score = min(1.0, (avg_charged_cluster_size * charged_cluster_count) / 10)
        hydrophobic_clustering_score = min(1.0, (avg_hydrophobic_cluster_size * hydrophobic_cluster_count) / 10)
        aromatic_clustering_score = min(1.0, (avg_aromatic_cluster_size * aromatic_cluster_count) / 10)
        
        # Calculate overall clustering score
        overall_clustering_score = (
            0.4 * charged_clustering_score +
            0.4 * hydrophobic_clustering_score +
            0.2 * aromatic_clustering_score
        )
        
        return {
            'charged_cluster_count': charged_cluster_count,
            'hydrophobic_cluster_count': hydrophobic_cluster_count,
            'aromatic_cluster_count': aromatic_cluster_count,
            'charged_cluster_density': charged_cluster_density,
            'hydrophobic_cluster_density': hydrophobic_cluster_density,
            'aromatic_cluster_density': aromatic_cluster_density,
            'avg_charged_cluster_size': avg_charged_cluster_size,
            'avg_hydrophobic_cluster_size': avg_hydrophobic_cluster_size,
            'avg_aromatic_cluster_size': avg_aromatic_cluster_size,
            'max_charged_cluster_size': max_charged_cluster_size,
            'max_hydrophobic_cluster_size': max_hydrophobic_cluster_size,
            'max_aromatic_cluster_size': max_aromatic_cluster_size,
            'charged_clustering_score': charged_clustering_score,
            'hydrophobic_clustering_score': hydrophobic_clustering_score,
            'aromatic_clustering_score': aromatic_clustering_score,
            'overall_clustering_score': overall_clustering_score
        }
    
    def calculate_clustering_risk_score(self, sequence: str) -> Dict[str, Union[float, str]]:
        """
        Calculate a comprehensive clustering risk score.
        
        Args:
            sequence (str): Amino acid sequence
            
        Returns:
            Dict: Clustering risk score and interpretation
        """
        # Analyze residue clustering
        clustering_analysis = self.analyze_residue_clustering(sequence)
        
        # Extract clustering score
        overall_clustering_score = clustering_analysis['cluster_metrics']['overall_clustering_score']
        
        # Interpret the score
        if overall_clustering_score < 0.1:
            interpretation = "Low clustering - favorable for solubility and stability"
        elif overall_clustering_score < 0.2:
            interpretation = "Moderate clustering - generally acceptable"
        elif overall_clustering_score < 0.3:
            interpretation = "High clustering - may affect solubility and stability"
        else:
            interpretation = "Very high clustering - likely to cause developability issues"
        
        return {
            'sequence': sequence,
            'clustering_risk_score': overall_clustering_score,
            'cluster_metrics': clustering_analysis['cluster_metrics'],
            'interpretation': interpretation,
            'scoring_complete': True
        }


def main():
    """
    Example usage of the residue clustering analyzer.
    """
    # Example sequence
    sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    
    # Create analyzer
    analyzer = ResidueClusteringAnalyzer()
    
    # Analyze residue clustering
    clustering_analysis = analyzer.analyze_residue_clustering(sequence)
    
    print("Residue Clustering Analysis:")
    print(f"  Sequence length: {clustering_analysis['length']}")
    print(f"  Charged clusters: {len(clustering_analysis['charged_clusters'])}")
    print(f"  Hydrophobic clusters: {len(clustering_analysis['hydrophobic_clusters'])}")
    print(f"  Aromatic clusters: {len(clustering_analysis['aromatic_clusters'])}")
    
    # Print cluster metrics
    metrics = clustering_analysis['cluster_metrics']
    print("\nCluster Metrics:")
    print(f"  Charged cluster density: {metrics['charged_cluster_density']:.3f}")
    print(f"  Hydrophobic cluster density: {metrics['hydrophobic_cluster_density']:.3f}")
    print(f"  Aromatic cluster density: {metrics['aromatic_cluster_density']:.3f}")
    print(f"  Average charged cluster size: {metrics['avg_charged_cluster_size']:.1f}")
    print(f"  Average hydrophobic cluster size: {metrics['avg_hydrophobic_cluster_size']:.1f}")
    print(f"  Average aromatic cluster size: {metrics['avg_aromatic_cluster_size']:.1f}")
    print(f"  Charged clustering score: {metrics['charged_clustering_score']:.3f}")
    print(f"  Hydrophobic clustering score: {metrics['hydrophobic_clustering_score']:.3f}")
    print(f"  Aromatic clustering score: {metrics['aromatic_clustering_score']:.3f}")
    print(f"  Overall clustering score: {metrics['overall_clustering_score']:.3f}")
    
    # Calculate clustering risk score
    risk_score = analyzer.calculate_clustering_risk_score(sequence)
    print("\nClustering Risk Score:")
    print(f"  Risk score: {risk_score['clustering_risk_score']:.3f}")
    print(f"  Interpretation: {risk_score['interpretation']}")


if __name__ == "__main__":
    main()
