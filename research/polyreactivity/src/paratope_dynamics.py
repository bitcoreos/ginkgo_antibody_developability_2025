"""
Paratope Dynamics Proxies Module

This module implements paratope dynamics proxies for antibody variable domains.
"""

import numpy as np
from typing import Dict, List, Tuple, Union
from collections import Counter

# Amino acid property groups relevant to paratope dynamics
PARATope_AA = 'YSGNFDHW'  # Tyrosine, Serine, Glycine, Asparagine, Phenylalanine, Aspartic acid, Histidine, Tryptophan
FLEXIBLE_AA = 'GS'  # Glycine, Serine
RIGID_AA = 'PW'  # Proline, Tryptophan


class ParatopeDynamicsAnalyzer:
    """
    Analyzer for paratope dynamics proxies in antibody variable domains.
    """
    
    def __init__(self):
        """
        Initialize the paratope dynamics analyzer.
        """
        pass
    
    def analyze_paratope_dynamics(self, sequence: str, cdr_regions: List[Tuple[int, int]] = None) -> Dict[str, Union[int, float, Dict]]:
        """
        Analyze paratope dynamics proxies in a sequence.
        
        Args:
            sequence (str): Amino acid sequence
            cdr_regions (List[Tuple[int, int]]): CDR regions as list of (start, end) tuples (0-indexed, inclusive)
            
        Returns:
            Dict: Paratope dynamics analysis results
        """
        sequence = sequence.upper()
        length = len(sequence)
        
        if length == 0:
            return {
                'sequence': sequence,
                'length': 0,
                'cdr_regions': cdr_regions or [],
                'paratope_dynamics': {},
                'dynamics_metrics': {},
                'analysis_complete': True
            }
        
        # If CDR regions not provided, use default CDR regions for antibody variable domains
        if cdr_regions is None:
            # Approximate CDR regions for a typical variable domain
            # CDR1: ~24-34, CDR2: ~50-56, CDR3: ~95-106
            cdr_regions = [(23, 33), (49, 55), (94, 105)]
        
        # Extract paratope regions (CDR regions)
        paratope_regions = self._extract_paratope_regions(sequence, cdr_regions)
        
        # Analyze paratope dynamics
        paratope_dynamics = self._analyze_paratope_dynamics(sequence, paratope_regions)
        
        # Calculate dynamics metrics
        dynamics_metrics = self._calculate_dynamics_metrics(sequence, paratope_dynamics)
        
        return {
            'sequence': sequence,
            'length': length,
            'cdr_regions': cdr_regions,
            'paratope_dynamics': paratope_dynamics,
            'dynamics_metrics': dynamics_metrics,
            'analysis_complete': True
        }
    
    def _extract_paratope_regions(self, sequence: str, cdr_regions: List[Tuple[int, int]]) -> List[str]:
        """
        Extract paratope regions (CDR sequences) from a sequence.
        
        Args:
            sequence (str): Amino acid sequence
            cdr_regions (List[Tuple[int, int]]): CDR regions as list of (start, end) tuples
            
        Returns:
            List[str]: List of CDR sequences
        """
        paratope_regions = []
        
        for start, end in cdr_regions:
            # Ensure indices are within sequence bounds
            start = max(0, start)
            end = min(len(sequence) - 1, end)
            
            if start <= end:
                paratope_regions.append(sequence[start:end+1])
        
        return paratope_regions
    
    def _analyze_paratope_dynamics(self, sequence: str, paratope_regions: List[str]) -> Dict[str, Union[int, float]]:
        """
        Analyze paratope dynamics proxies.
        
        Args:
            sequence (str): Amino acid sequence
            paratope_regions (List[str]): Paratope (CDR) regions
            
        Returns:
            Dict: Paratope dynamics analysis
        """
        # Analyze each paratope region
        paratope_analyses = []
        
        for i, region in enumerate(paratope_regions):
            region_analysis = self._analyze_region_dynamics(region)
            region_analysis['region_index'] = i
            paratope_analyses.append(region_analysis)
        
        return {
            'paratope_regions': paratope_regions,
            'region_analyses': paratope_analyses,
            'region_count': len(paratope_regions)
        }
    
    def _analyze_region_dynamics(self, region: str) -> Dict[str, Union[int, float]]:
        """
        Analyze dynamics of a single region.
        
        Args:
            region (str): Amino acid sequence of region
            
        Returns:
            Dict: Region dynamics analysis
        """
        length = len(region)
        
        if length == 0:
            return {
                'length': 0,
                'flexible_residue_count': 0,
                'rigid_residue_count': 0,
                'paratope_residue_count': 0,
                'flexibility_score': 0.0,
                'rigidity_score': 0.0,
                'paratope_score': 0.0,
                'entropy_proxy': 0.0
            }
        
        # Count residue types
        flexible_count = sum(1 for aa in region if aa in FLEXIBLE_AA)
        rigid_count = sum(1 for aa in region if aa in RIGID_AA)
        paratope_count = sum(1 for aa in region if aa in PARATope_AA)
        
        # Calculate scores
        flexibility_score = flexible_count / length
        rigidity_score = rigid_count / length
        paratope_score = paratope_count / length
        
        # Calculate entropy proxy (simplified)
        # Higher entropy = more diverse residue composition
        residue_counts = Counter(region)
        residue_frequencies = [count / length for count in residue_counts.values()]
        entropy_proxy = -sum(p * np.log(p) for p in residue_frequencies if p > 0)
        
        return {
            'length': length,
            'flexible_residue_count': flexible_count,
            'rigid_residue_count': rigid_count,
            'paratope_residue_count': paratope_count,
            'flexibility_score': flexibility_score,
            'rigidity_score': rigidity_score,
            'paratope_score': paratope_score,
            'entropy_proxy': entropy_proxy
        }
    
    def _calculate_dynamics_metrics(self, sequence: str, paratope_dynamics: Dict) -> Dict[str, Union[int, float]]:
        """
        Calculate overall dynamics metrics.
        
        Args:
            sequence (str): Amino acid sequence
            paratope_dynamics (Dict): Paratope dynamics analysis
            
        Returns:
            Dict: Dynamics metrics
        """
        length = len(sequence)
        
        if length == 0:
            return {}
        
        # Extract region analyses
        region_analyses = paratope_dynamics['region_analyses']
        
        if not region_analyses:
            return {
                'avg_flexibility_score': 0.0,
                'avg_rigidity_score': 0.0,
                'avg_paratope_score': 0.0,
                'avg_entropy_proxy': 0.0,
                'dynamics_score': 0.0
            }
        
        # Calculate average scores across regions
        avg_flexibility_score = np.mean([r['flexibility_score'] for r in region_analyses])
        avg_rigidity_score = np.mean([r['rigidity_score'] for r in region_analyses])
        avg_paratope_score = np.mean([r['paratope_score'] for r in region_analyses])
        avg_entropy_proxy = np.mean([r['entropy_proxy'] for r in region_analyses])
        
        # Calculate overall dynamics score (0-1, higher means more dynamic)
        # Weight flexibility and entropy more heavily
        dynamics_score = (
            0.4 * avg_flexibility_score +
            0.2 * avg_rigidity_score +
            0.2 * avg_paratope_score +
            0.2 * min(1.0, avg_entropy_proxy / 5.0)  # Normalize entropy
        )
        
        return {
            'avg_flexibility_score': avg_flexibility_score,
            'avg_rigidity_score': avg_rigidity_score,
            'avg_paratope_score': avg_paratope_score,
            'avg_entropy_proxy': avg_entropy_proxy,
            'dynamics_score': dynamics_score
        }
    
    def calculate_dynamics_risk_score(self, sequence: str, cdr_regions: List[Tuple[int, int]] = None) -> Dict[str, Union[float, str]]:
        """
        Calculate a comprehensive dynamics risk score.
        
        Args:
            sequence (str): Amino acid sequence
            cdr_regions (List[Tuple[int, int]]): CDR regions as list of (start, end) tuples
            
        Returns:
            Dict: Dynamics risk score and interpretation
        """
        # Analyze paratope dynamics
        dynamics_analysis = self.analyze_paratope_dynamics(sequence, cdr_regions)
        
        # Extract dynamics score
        dynamics_score = dynamics_analysis['dynamics_metrics']['dynamics_score']
        
        # Interpret the score
        if dynamics_score < 0.2:
            interpretation = "Low dynamics - may limit binding flexibility"
        elif dynamics_score < 0.4:
            interpretation = "Moderate dynamics - generally favorable for binding"
        elif dynamics_score < 0.6:
            interpretation = "High dynamics - may affect binding specificity"
        else:
            interpretation = "Very high dynamics - likely to cause binding instability"
        
        return {
            'sequence': sequence,
            'cdr_regions': cdr_regions,
            'dynamics_risk_score': dynamics_score,
            'dynamics_metrics': dynamics_analysis['dynamics_metrics'],
            'interpretation': interpretation,
            'scoring_complete': True
        }


def main():
    """
    Example usage of the paratope dynamics analyzer.
    """
    # Example sequence
    sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    
    # Define CDR regions (approximate)
    cdr_regions = [(23, 33), (49, 55), (94, 105)]
    
    # Create analyzer
    analyzer = ParatopeDynamicsAnalyzer()
    
    # Analyze paratope dynamics
    dynamics_analysis = analyzer.analyze_paratope_dynamics(sequence, cdr_regions)
    
    print("Paratope Dynamics Analysis:")
    print(f"  Sequence length: {dynamics_analysis['length']}")
    print(f"  Number of CDR regions: {dynamics_analysis['paratope_dynamics']['region_count']}")
    
    # Print dynamics metrics
    metrics = dynamics_analysis['dynamics_metrics']
    print("\nDynamics Metrics:")
    print(f"  Average flexibility score: {metrics['avg_flexibility_score']:.3f}")
    print(f"  Average rigidity score: {metrics['avg_rigidity_score']:.3f}")
    print(f"  Average paratope score: {metrics['avg_paratope_score']:.3f}")
    print(f"  Average entropy proxy: {metrics['avg_entropy_proxy']:.3f}")
    print(f"  Dynamics score: {metrics['dynamics_score']:.3f}")
    
    # Calculate dynamics risk score
    risk_score = analyzer.calculate_dynamics_risk_score(sequence, cdr_regions)
    print("\nDynamics Risk Score:")
    print(f"  Risk score: {risk_score['dynamics_risk_score']:.3f}")
    print(f"  Interpretation: {risk_score['interpretation']}")


if __name__ == "__main__":
    main()
