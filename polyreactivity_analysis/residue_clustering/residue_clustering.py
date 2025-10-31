"""
Residue Clustering Pattern Analysis Implementation
"""

import numpy as np
from collections import defaultdict

class ResidueClusteringAnalyzer:
    """
    Implementation of Residue Clustering Pattern Analysis
    """
    
    def __init__(self):
        """
        Initialize Residue Clustering Analyzer
        """
        # Define amino acid property groups
        self.charged_aas = set(['K', 'R', 'H', 'D', 'E'])  # Lysine, Arginine, Histidine, Aspartic acid, Glutamic acid
        self.hydrophobic_aas = set(['A', 'I', 'L', 'M', 'F', 'P', 'W', 'V'])  # Alanine, Isoleucine, Leucine, Methionine, Phenylalanine, Proline, Tryptophan, Valine
        self.aromatic_aas = set(['F', 'W', 'Y'])  # Phenylalanine, Tryptophan, Tyrosine
        
        # Combine all property groups for clustering analysis
        self.all_property_groups = {
            'charged': self.charged_aas,
            'hydrophobic': self.hydrophobic_aas,
            'aromatic': self.aromatic_aas
        }
    
    def _identify_clusters(self, sequence, property_group):
        """
        Identify clusters of consecutive residues belonging to a property group
        
        Parameters:
        sequence (str): Amino acid sequence
        property_group (set): Set of amino acids belonging to a property group
        
        Returns:
        list: List of tuples (start_position, end_position) for each cluster
        """
        clusters = []
        in_cluster = False
        cluster_start = 0
        
        for i, aa in enumerate(sequence):
            if aa in property_group:
                if not in_cluster:
                    # Start of a new cluster
                    in_cluster = True
                    cluster_start = i
            else:
                if in_cluster:
                    # End of current cluster
                    clusters.append((cluster_start, i-1))
                    in_cluster = False
        
        # Handle case where sequence ends with a cluster
        if in_cluster:
            clusters.append((cluster_start, len(sequence)-1))
        
        return clusters
    
    def _calculate_cluster_metrics(self, clusters, sequence_length):
        """
        Calculate cluster metrics
        
        Parameters:
        clusters (list): List of tuples (start_position, end_position) for each cluster
        sequence_length (int): Length of the sequence
        
        Returns:
        dict: Cluster metrics
        """
        cluster_count = len(clusters)
        cluster_density = cluster_count / sequence_length if sequence_length > 0 else 0
        
        if cluster_count > 0:
            cluster_sizes = [end - start + 1 for start, end in clusters]
            avg_cluster_size = np.mean(cluster_sizes)
            max_cluster_size = np.max(cluster_sizes)
        else:
            avg_cluster_size = 0
            max_cluster_size = 0
        
        return {
            'cluster_count': cluster_count,
            'cluster_density': cluster_density,
            'avg_cluster_size': avg_cluster_size,
            'max_cluster_size': max_cluster_size
        }
    
    def analyze_residue_clustering(self, sequence):
        """
        Analyze residue clustering patterns in a sequence
        
        Parameters:
        sequence (str): Amino acid sequence
        
        Returns:
        dict: Clustering analysis results
        """
        sequence_length = len(sequence)
        results = {}
        
        # Analyze each property group
        for group_name, property_group in self.all_property_groups.items():
            # Identify clusters
            clusters = self._identify_clusters(sequence, property_group)
            
            # Calculate metrics
            metrics = self._calculate_cluster_metrics(clusters, sequence_length)
            
            # Calculate clustering score (0-1, higher means more clustering)
            # Using the formula from the documentation: min(1.0, (avg_cluster_size * cluster_count) / 10)
            if metrics['cluster_count'] > 0:
                clustering_score = min(1.0, (metrics['avg_cluster_size'] * metrics['cluster_count']) / 10)
            else:
                clustering_score = 0
            
            results[group_name] = {
                'clusters': clusters,
                'metrics': metrics,
                'clustering_score': clustering_score
            }
        
        return results
    
    def calculate_clustering_risk_score(self, sequence):
        """
        Calculate a comprehensive clustering risk score
        
        Parameters:
        sequence (str): Amino acid sequence
        
        Returns:
        dict: Clustering risk score and interpretation
        """
        # Analyze residue clustering
        clustering_results = self.analyze_residue_clustering(sequence)
        
        # Extract clustering scores
        charged_clustering_score = clustering_results['charged']['clustering_score']
        hydrophobic_clustering_score = clustering_results['hydrophobic']['clustering_score']
        aromatic_clustering_score = clustering_results['aromatic']['clustering_score']
        
        # Calculate overall clustering score using weighted approach
        # Formula from documentation: 0.4 * charged + 0.4 * hydrophobic + 0.2 * aromatic
        overall_clustering_score = (
            0.4 * charged_clustering_score +
            0.4 * hydrophobic_clustering_score +
            0.2 * aromatic_clustering_score
        )
        
        # Interpretation
        if overall_clustering_score < 0.1:
            interpretation = "Low clustering - favorable for solubility and stability"
        elif overall_clustering_score < 0.2:
            interpretation = "Moderate clustering - generally acceptable"
        elif overall_clustering_score < 0.3:
            interpretation = "High clustering - may affect solubility and stability"
        else:
            interpretation = "Very high clustering - likely to cause developability issues"
        
        return {
            'clustering_results': clustering_results,
            'clustering_risk_score': overall_clustering_score,
            'charged_clustering_score': charged_clustering_score,
            'hydrophobic_clustering_score': hydrophobic_clustering_score,
            'aromatic_clustering_score': aromatic_clustering_score,
            'interpretation': interpretation
        }

def main():
    """
    Main function for testing Residue Clustering Analyzer
    """
    print("Testing Residue Clustering Analyzer...")
    
    # Example sequence
    sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    
    # Create analyzer
    analyzer = ResidueClusteringAnalyzer()
    
    # Analyze residue clustering
    clustering_results = analyzer.analyze_residue_clustering(sequence)
    print(f"Charged residue clusters: {clustering_results['charged']['clusters']}")
    print(f"Hydrophobic residue clusters: {clustering_results['hydrophobic']['clusters']}")
    print(f"Aromatic residue clusters: {clustering_results['aromatic']['clusters']}")
    
    # Calculate clustering risk score
    risk_score = analyzer.calculate_clustering_risk_score(sequence)
    print(f"\nClustering risk score: {risk_score['clustering_risk_score']:.3f}")
    print(f"Interpretation: {risk_score['interpretation']}")
    
    print("\nResidue Clustering Analyzer test completed successfully!")


if __name__ == '__main__':
    main()
