"""
Test script for FragmentAnalyzer
"""

import sys
import os

# Add the fragment_analyzer directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fragment_analyzer'))

# Import the FragmentAnalyzer class directly
from fragment_analyzer import FragmentAnalyzer

def test_fragment_analyzer():
    """
    Test the FragmentAnalyzer class.
    """
    # Create an instance of FragmentAnalyzer
    analyzer = FragmentAnalyzer()
    
    # Test sequence
    test_sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDFWSGYWGQGTLVTVSS"
    
    print("Testing FragmentAnalyzer with sample sequence:")
    print(f"Sequence: {test_sequence}")
    print(f"Sequence length: {len(test_sequence)}")
    
    # Test sequence analysis
    seq_results = analyzer.analyze_sequence(test_sequence)
    print(f"\n=== SEQUENCE ANALYSIS ===")
    print(f"Length: {seq_results['length']}")
    
    # Print composition results
    composition = seq_results['composition']
    print(f"\nAmino acid composition:")
    print(f"  Total amino acids: {composition['total_amino_acids']}")
    print(f"  Hydrophobic: {composition['groups']['hydrophobic']} ({composition['group_percentages']['hydrophobic']:.2f}%)")
    print(f"  Charged: {composition['groups']['charged']} ({composition['group_percentages']['charged']:.2f}%)")
    print(f"  Polar: {composition['groups']['polar']} ({composition['group_percentages']['polar']:.2f}%)")
    print(f"  Aromatic: {composition['groups']['aromatic']} ({composition['group_percentages']['aromatic']:.2f}%)")
    
    # Print complexity results
    complexity = seq_results['complexity']
    print(f"\nSequence complexity:")
    print(f"  Overall complexity: {complexity['overall_complexity']:.4f}")
    print(f"  K-mer diversities:")
    for k in [2, 3, 4]:
        if f'k{k}_diversity' in complexity:
            print(f"    k={k}: {complexity[f'k{k}_diversity']:.4f}")
    
    # Test physicochemical properties
    phys_props = analyzer.calculate_physicochemical_properties(test_sequence)
    print(f"\n=== PHYSICOCHEMICAL PROPERTIES ===")
    
    # Charge distribution
    charge_dist = phys_props['charge_distribution']
    print(f"Charge distribution:")
    print(f"  Positive charges: {charge_dist['positive_charges']}")
    print(f"  Negative charges: {charge_dist['negative_charges']}")
    print(f"  Net charge: {charge_dist['net_charge']}")
    print(f"  Charge density: {charge_dist['charge_density']:.4f}")
    
    # Hydrophobicity
    hydrophobicity = phys_props['hydrophobicity']
    print(f"Hydrophobicity: {hydrophobicity:.4f}")
    
    # Isoelectric point
    pi = phys_props['isoelectric_point']
    print(f"Isoelectric point: {pi:.2f}")
    
    # Test stability assessment
    stability = analyzer.assess_stability(test_sequence)
    print(f"\n=== STABILITY ASSESSMENT ===")
    print(f"Thermal stability: {stability['thermal_stability']:.4f}")
    print(f"Aggregation propensity: {stability['aggregation_propensity']:.4f}")

if __name__ == "__main__":
    test_fragment_analyzer()
