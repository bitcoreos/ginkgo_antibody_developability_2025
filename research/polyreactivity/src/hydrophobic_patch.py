"""
Hydrophobic Patch Analysis Module

This module implements hydrophobic patch analysis for antibody variable domains.
"""

import numpy as np
from typing import Dict, List, Tuple, Union
from collections import Counter

# Hydrophobic amino acids
HYDROPHOBIC_AA = 'AILMFPWV'  # Alanine, Isoleucine, Leucine, Methionine, Phenylalanine, Proline, Tryptophan, Valine


class HydrophobicPatchAnalyzer:
    """
    Analyzer for hydrophobic patches in antibody variable domains.
    """
    
    def __init__(self):
        """
        Initialize the hydrophobic patch analyzer.
        """
        pass
    
    def analyze_hydrophobic_patches(self, sequence: str, window_size: int = 5) -> Dict[str, Union[int, float, List]]:
        """
        Analyze hydrophobic patches in a sequence.
        
        Args:
            sequence (str): Amino acid sequence
            window_size (int): Size of sliding window for patch detection (default: 5)
            
        Returns:
            Dict: Hydrophobic patch analysis results
        """
        sequence = sequence.upper()
        length = len(sequence)
        
        if length == 0:
            return {
                'sequence': sequence,
                'length': 0,
                'window_size': window_size,
                'hydrophobic_patches': [],
                'patch_metrics': {},
                'analysis_complete': True
            }
        
        # Find hydrophobic patches using sliding window
        hydrophobic_patches = self._find_hydrophobic_patches(sequence, window_size)
        
        # Calculate patch metrics
        patch_metrics = self._calculate_patch_metrics(sequence, hydrophobic_patches, window_size)
        
        return {
            'sequence': sequence,
            'length': length,
            'window_size': window_size,
            'hydrophobic_patches': hydrophobic_patches,
            'patch_metrics': patch_metrics,
            'analysis_complete': True
        }
    
    def _find_hydrophobic_patches(self, sequence: str, window_size: int) -> List[Dict]:
        """
        Find hydrophobic patches in a sequence using sliding window.
        
        Args:
            sequence (str): Amino acid sequence
            window_size (int): Size of sliding window
            
        Returns:
            List[Dict]: List of hydrophobic patch information
        """
        patches = []
        
        # Use sliding window to find patches
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]
            hydrophobic_count = sum(1 for aa in window if aa in HYDROPHOBIC_AA)
            
            # Consider it a patch if more than half of the residues are hydrophobic
            if hydrophobic_count > window_size / 2:
                patches.append({
                    'start': i,
                    'end': i + window_size - 1,
                    'window': window,
                    'hydrophobic_count': hydrophobic_count,
                    'hydrophobic_fraction': hydrophobic_count / window_size
                })
        
        return patches
    
    def _calculate_patch_metrics(self, sequence: str, patches: List[Dict], window_size: int) -> Dict[str, Union[int, float]]:
        """
        Calculate metrics for hydrophobic patches.
        
        Args:
            sequence (str): Amino acid sequence
            patches (List[Dict]): Hydrophobic patches
            window_size (int): Size of sliding window
            
        Returns:
            Dict: Patch metrics
        """
        length = len(sequence)
        
        if length == 0:
            return {}
        
        # Count patches
        patch_count = len(patches)
        
        # Calculate patch density
        patch_density = patch_count / max(1, length - window_size + 1)
        
        # Calculate average hydrophobic fraction
        avg_hydrophobic_fraction = np.mean([p['hydrophobic_fraction'] for p in patches]) if patches else 0
        
        # Calculate maximum hydrophobic fraction
        max_hydrophobic_fraction = max([p['hydrophobic_fraction'] for p in patches]) if patches else 0
        
        # Calculate patch score (0-1, higher means more hydrophobic patches)
        # Normalize by maximum possible patches
        max_possible_patches = max(1, length - window_size + 1)
        patch_score = min(1.0, patch_count / max_possible_patches)
        
        # Calculate hydrophobicity score (0-1, higher means more hydrophobic)
        total_hydrophobic = sum(1 for aa in sequence if aa in HYDROPHOBIC_AA)
        hydrophobicity_score = total_hydrophobic / length if length > 0 else 0
        
        return {
            'patch_count': patch_count,
            'patch_density': patch_density,
            'avg_hydrophobic_fraction': avg_hydrophobic_fraction,
            'max_hydrophobic_fraction': max_hydrophobic_fraction,
            'patch_score': patch_score,
            'hydrophobicity_score': hydrophobicity_score
        }
    
    def calculate_binding_potential(self, sequence: str, window_size: int = 5) -> Dict[str, Union[float, str]]:
        """
        Calculate hydrophobic patch binding potential.
        
        Args:
            sequence (str): Amino acid sequence
            window_size (int): Size of sliding window for patch detection (default: 5)
            
        Returns:
            Dict: Binding potential score and interpretation
        """
        # Analyze hydrophobic patches
        patch_analysis = self.analyze_hydrophobic_patches(sequence, window_size)
        
        # Extract relevant metrics
        patch_score = patch_analysis['patch_metrics']['patch_score']
        hydrophobicity_score = patch_analysis['patch_metrics']['hydrophobicity_score']
        avg_hydrophobic_fraction = patch_analysis['patch_metrics']['avg_hydrophobic_fraction']
        
        # Calculate binding potential score
        # Weight patch score more heavily as it represents the clustering of hydrophobic residues
        binding_potential = (
            0.5 * patch_score +
            0.3 * hydrophobicity_score +
            0.2 * avg_hydrophobic_fraction
        )
        
        # Interpret the score
        if binding_potential < 0.2:
            interpretation = "Low binding potential - favorable for solubility and specificity"
        elif binding_potential < 0.4:
            interpretation = "Moderate binding potential - generally acceptable"
        elif binding_potential < 0.6:
            interpretation = "High binding potential - may affect solubility and specificity"
        else:
            interpretation = "Very high binding potential - likely to cause non-specific binding"
        
        return {
            'sequence': sequence,
            'window_size': window_size,
            'binding_potential': binding_potential,
            'patch_score': patch_score,
            'hydrophobicity_score': hydrophobicity_score,
            'avg_hydrophobic_fraction': avg_hydrophobic_fraction,
            'interpretation': interpretation,
            'scoring_complete': True
        }


def main():
    """
    Example usage of the hydrophobic patch analyzer.
    """
    # Example sequence
    sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    
    # Create analyzer
    analyzer = HydrophobicPatchAnalyzer()
    
    # Analyze hydrophobic patches
    patch_analysis = analyzer.analyze_hydrophobic_patches(sequence)
    
    print("Hydrophobic Patch Analysis:")
    print(f"  Sequence length: {patch_analysis['length']}")
    print(f"  Window size: {patch_analysis['window_size']}")
    print(f"  Number of patches: {len(patch_analysis['hydrophobic_patches'])}")
    
    # Print patch metrics
    metrics = patch_analysis['patch_metrics']
    print("\nPatch Metrics:")
    print(f"  Patch density: {metrics['patch_density']:.3f}")
    print(f"  Average hydrophobic fraction: {metrics['avg_hydrophobic_fraction']:.3f}")
    print(f"  Maximum hydrophobic fraction: {metrics['max_hydrophobic_fraction']:.3f}")
    print(f"  Patch score: {metrics['patch_score']:.3f}")
    print(f"  Hydrophobicity score: {metrics['hydrophobicity_score']:.3f}")
    
    # Calculate binding potential
    binding_potential = analyzer.calculate_binding_potential(sequence)
    print("\nBinding Potential:")
    print(f"  Binding potential: {binding_potential['binding_potential']:.3f}")
    print(f"  Patch score: {binding_potential['patch_score']:.3f}")
    print(f"  Hydrophobicity score: {binding_potential['hydrophobicity_score']:.3f}")
    print(f"  Average hydrophobic fraction: {binding_potential['avg_hydrophobic_fraction']:.3f}")
    print(f"  Interpretation: {binding_potential['interpretation']}")


if __name__ == "__main__":
    main()
