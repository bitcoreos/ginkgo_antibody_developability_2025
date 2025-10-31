"""
Paratope Dynamics Proxies Implementation
"""

import numpy as np
from collections import Counter

class ParatopeDynamicsAnalyzer:
    """
    Implementation of Paratope Dynamics Proxies
    """
    
    def __init__(self):
        """
        Initialize Paratope Dynamics Analyzer
        """
        # Define amino acid property groups relevant to paratope dynamics
        self.paratope_aas = set(['Y', 'S', 'G', 'N', 'F', 'D', 'H', 'W'])  # Tyrosine, Serine, Glycine, Asparagine, Phenylalanine, Aspartic acid, Histidine, Tryptophan
        self.flexible_aas = set(['G', 'S'])  # Glycine, Serine
        self.rigid_aas = set(['P', 'W'])  # Proline, Tryptophan
        
        # Default approximate CDR regions (CDR1: ~24-34, CDR2: ~50-56, CDR3: ~95-106)
        self.default_cdr_regions = [(23, 33), (49, 55), (94, 105)]
    
    def _calculate_entropy_proxy(self, sequence):
        """
        Calculate a simplified entropy proxy based on residue composition diversity
        
        Parameters:
        sequence (str): Amino acid sequence
        
        Returns:
        float: Entropy proxy value
        """
        if len(sequence) == 0:
            return 0
        
        # Count residue frequencies
        residue_counts = Counter(sequence)
        sequence_length = len(sequence)
        
        # Calculate entropy proxy
        entropy_proxy = 0
        for count in residue_counts.values():
            frequency = count / sequence_length
            if frequency > 0:
                entropy_proxy -= frequency * np.log(frequency)
        
        return entropy_proxy
    
    def _analyze_cdr_region(self, cdr_sequence):
        """
        Analyze a CDR region for dynamics properties
        
        Parameters:
        cdr_sequence (str): CDR region sequence
        
        Returns:
        dict: Dynamics properties for the CDR region
        """
        sequence_length = len(cdr_sequence)
        
        if sequence_length == 0:
            return {
                'flexible_count': 0,
                'flexibility_score': 0,
                'rigid_count': 0,
                'rigidity_score': 0,
                'paratope_count': 0,
                'paratope_score': 0,
                'entropy_proxy': 0
            }
        
        # Count flexible residues
        flexible_count = sum(1 for aa in cdr_sequence if aa in self.flexible_aas)
        flexibility_score = flexible_count / sequence_length
        
        # Count rigid residues
        rigid_count = sum(1 for aa in cdr_sequence if aa in self.rigid_aas)
        rigidity_score = rigid_count / sequence_length
        
        # Count paratope residues
        paratope_count = sum(1 for aa in cdr_sequence if aa in self.paratope_aas)
        paratope_score = paratope_count / sequence_length
        
        # Calculate entropy proxy
        entropy_proxy = self._calculate_entropy_proxy(cdr_sequence)
        
        return {
            'flexible_count': flexible_count,
            'flexibility_score': flexibility_score,
            'rigid_count': rigid_count,
            'rigidity_score': rigidity_score,
            'paratope_count': paratope_count,
            'paratope_score': paratope_score,
            'entropy_proxy': entropy_proxy
        }
    
    def analyze_paratope_dynamics(self, sequence, cdr_regions=None):
        """
        Analyze paratope dynamics proxies in a sequence
        
        Parameters:
        sequence (str): Amino acid sequence
        cdr_regions (list): List of tuples (start, end) for CDR regions
        
        Returns:
        dict: Paratope dynamics analysis results
        """
        # Use default CDR regions if not provided
        if cdr_regions is None:
            cdr_regions = self.default_cdr_regions
        
        # Extract CDR sequences
        cdr_sequences = []
        for start, end in cdr_regions:
            # Adjust for 0-based indexing and ensure we don't go out of bounds
            start = max(0, start)
            end = min(len(sequence), end + 1)  # +1 because slicing is exclusive of end
            cdr_sequences.append(sequence[start:end])
        
        # Analyze each CDR region
        cdr_results = []
        for i, cdr_sequence in enumerate(cdr_sequences):
            cdr_analysis = self._analyze_cdr_region(cdr_sequence)
            cdr_analysis['cdr_number'] = i + 1
            cdr_results.append(cdr_analysis)
        
        return {
            'cdr_regions': cdr_regions,
            'cdr_sequences': cdr_sequences,
            'cdr_results': cdr_results
        }
    
    def calculate_dynamics_risk_score(self, sequence, cdr_regions=None):
        """
        Calculate a comprehensive dynamics risk score
        
        Parameters:
        sequence (str): Amino acid sequence
        cdr_regions (list): List of tuples (start, end) for CDR regions
        
        Returns:
        dict: Dynamics risk score and interpretation
        """
        # Analyze paratope dynamics
        dynamics_results = self.analyze_paratope_dynamics(sequence, cdr_regions)
        
        # Extract scores from all CDR regions
        flexibility_scores = [cdr['flexibility_score'] for cdr in dynamics_results['cdr_results']]
        rigidity_scores = [cdr['rigidity_score'] for cdr in dynamics_results['cdr_results']]
        paratope_scores = [cdr['paratope_score'] for cdr in dynamics_results['cdr_results']]
        entropy_proxies = [cdr['entropy_proxy'] for cdr in dynamics_results['cdr_results']]
        
        # Calculate average scores
        avg_flexibility_score = np.mean(flexibility_scores) if flexibility_scores else 0
        avg_rigidity_score = np.mean(rigidity_scores) if rigidity_scores else 0
        avg_paratope_score = np.mean(paratope_scores) if paratope_scores else 0
        avg_entropy_proxy = np.mean(entropy_proxies) if entropy_proxies else 0
        
        # Calculate overall dynamics score using weighted approach
        # Formula from documentation: 0.4 * avg_flexibility + 0.2 * avg_rigidity + 0.2 * avg_paratope + 0.2 * normalized_entropy
        normalized_entropy = min(1.0, avg_entropy_proxy / 5.0)  # Normalize entropy
        dynamics_score = (
            0.4 * avg_flexibility_score +
            0.2 * avg_rigidity_score +
            0.2 * avg_paratope_score +
            0.2 * normalized_entropy
        )
        
        # Interpretation
        if dynamics_score < 0.2:
            interpretation = "Low dynamics - may limit binding flexibility"
        elif dynamics_score < 0.4:
            interpretation = "Moderate dynamics - generally favorable for binding"
        elif dynamics_score < 0.6:
            interpretation = "High dynamics - may affect binding specificity"
        else:
            interpretation = "Very high dynamics - likely to cause binding instability"
        
        return {
            'dynamics_results': dynamics_results,
            'dynamics_risk_score': dynamics_score,
            'avg_flexibility_score': avg_flexibility_score,
            'avg_rigidity_score': avg_rigidity_score,
            'avg_paratope_score': avg_paratope_score,
            'avg_entropy_proxy': avg_entropy_proxy,
            'interpretation': interpretation
        }

def main():
    """
    Main function for testing Paratope Dynamics Analyzer
    """
    print("Testing Paratope Dynamics Analyzer...")
    
    # Example sequence
    sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    
    # Define CDR regions (approximate)
    cdr_regions = [(23, 33), (49, 55), (94, 105)]
    
    # Create analyzer
    analyzer = ParatopeDynamicsAnalyzer()
    
    # Analyze paratope dynamics
    dynamics_results = analyzer.analyze_paratope_dynamics(sequence, cdr_regions)
    print(f"CDR regions: {dynamics_results['cdr_regions']}")
    print(f"CDR sequences: {dynamics_results['cdr_sequences']}")
    
    # Calculate dynamics risk score
    dynamics_risk = analyzer.calculate_dynamics_risk_score(sequence, cdr_regions)
    print(f"\nDynamics risk score: {dynamics_risk['dynamics_risk_score']:.3f}")
    print(f"Interpretation: {dynamics_risk['interpretation']}")
    
    print("\nParatope Dynamics Analyzer test completed successfully!")


if __name__ == '__main__':
    main()
