"""
Comprehensive test script for FLAb Framework

This script demonstrates the complete workflow of the FLAb framework.
"""

import sys
import os
import json

# Add paths for polyreactivity analysis and validation systems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, '/a0/bitcore/workspace/polyreactivity_analysis')
# sys.path.insert(0, '/a0/bitcore/workspace/validation_systems')
# sys.path.insert(0, '/a0/bitcore/workspace/flab_framework/propemab')

# Import all FLAb modules directly from their Python files
from fragment_analyzer.fragment_analyzer import FragmentAnalyzer
from fragment_database import FragmentDatabase
from developability_predictor import DevelopabilityPredictor
from optimization_recommender import OptimizationRecommender

# Import polyreactivity analysis
from flab_integration import FLAbPolyreactivityAnalyzer

# Import validation systems
from validation_framework import SystematicValidationProtocol, ConceptDriftDetector, SubmissionQualityAssurance, ProspectiveValidationFramework

# Import PROPERMAB
from flab_propemab import FLAbPROPERMAB

def test_flab_framework():
    """
    Test the complete FLAb framework workflow.
    """
    print("=== FLAb Framework Comprehensive Test ===\n")

    # Test sequence
    test_sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDFWSGYWGQGTLVTVSS"
    fragment_id = "test_fragment_001"

    print(f"Analyzing antibody fragment: {fragment_id}")
    print(f"Sequence length: {len(test_sequence)}\n")

    # Step 1: Fragment Analysis
    print("Step 1: Fragment Analysis")
    analyzer = FragmentAnalyzer()
    fragment_analysis = analyzer.analyze_sequence(test_sequence)
    phys_props = analyzer.calculate_physicochemical_properties(test_sequence)
    stability = analyzer.assess_stability(test_sequence)

    # Combine all analysis results
    full_analysis = {
        'sequence': test_sequence,
        'composition': fragment_analysis['composition'],
        'complexity': fragment_analysis['complexity'],
        'physicochemical_properties': phys_props,
        'stability': stability
    }

    print(f"  - Amino acid composition: {fragment_analysis['composition']['total_amino_acids']} residues")
    print(f"  - Sequence complexity: {fragment_analysis['complexity']['overall_complexity']:.4f}")
    print(f"  - Isoelectric point: {phys_props['isoelectric_point']:.2f}")
    print(f"  - Hydrophobicity: {phys_props['hydrophobicity']:.4f}")
    print(f"  - Aggregation propensity: {stability['aggregation_propensity']:.4f}")
    print(f"  - Thermal stability: {stability['thermal_stability']:.4f}\n")

    # Step 2: Fragment Database Storage
    print("Step 2: Fragment Database Storage")
    db = FragmentDatabase('test_fragment_database.json')
    db.add_fragment(fragment_id, full_analysis)
    stored_fragment = db.get_fragment(fragment_id)
    print(f"  - Fragment stored successfully: {stored_fragment is not None}\n")

    # Step 3: Developability Prediction
    print("Step 3: Developability Prediction")
    # Load fragment database for PROPERMAB training
    fragment_database = db.load_database()

    # Initialize and train PROPERMAB
    propemab = FLAbPROPERMAB()
    propemab.train(fragment_database)

    # Predict developability using PROPERMAB
    propemab_predictions = propemab.predict_developability(full_analysis)
    print(f"  - Solubility prediction (PROPERMAB): {propemab_predictions['solubility']:.4f}")
    print(f"  - Expression level prediction (PROPERMAB): {propemab_predictions['expression_level']:.4f}")
    print(f"  - Aggregation propensity prediction (PROPERMAB): {propemab_predictions['aggregation_propensity']:.4f}")
    print(f"  - Thermal stability prediction (PROPERMAB): {propemab_predictions['thermal_stability']:.4f}")
    print(f"  - Immunogenicity prediction (PROPERMAB): {propemab_predictions['immunogenicity']:.4f}")

    # Predict developability using DevelopabilityPredictor with Multi-Channel Information Theory integration
    predictor = DevelopabilityPredictor()
    developability_predictions = predictor.predict_developability(full_analysis)
    print(f"  - Solubility prediction (DevelopabilityPredictor): {developability_predictions['solubility']:.4f}")
    print(f"  - Expression level prediction (DevelopabilityPredictor): {developability_predictions['expression']:.4f}")
    print(f"  - Aggregation propensity prediction (DevelopabilityPredictor): {developability_predictions['aggregation']:.4f}")
    print(f"  - Immunogenicity prediction (DevelopabilityPredictor): {developability_predictions['immunogenicity']:.4f}")
    print(f"  - Overall developability score (DevelopabilityPredictor): {developability_predictions['overall_score']:.4f}")
    print(f"  - Information theory analysis: {developability_predictions.get('information_theory_analysis', 'N/A')}")
    print(f"  - Information theoretic features: {list(developability_predictions.get('information_theoretic_features', {}).keys()) if developability_predictions.get('information_theoretic_features') else 'N/A'}")
    print()

    # Step 4: Optimization Recommendations
    print("Step 4: Optimization Recommendations")
    recommender = OptimizationRecommender()
    recommendations = recommender.generate_recommendations(full_analysis, propemab_predictions)
    print(f"  - Priority: {recommendations['priority']}")
    print(f"  - Sequence modifications: {len(recommendations['sequence_modifications'])}")
    print(f"  - Structural strategies: {len(recommendations['structural_strategies'])}")
    print(f"  - Design improvements: {len(recommendations['design_improvements'])}")

    # Print first recommendation if any exist
    all_recommendations = (
        recommendations['sequence_modifications'] +
        recommendations['structural_strategies'] +
        recommendations['design_improvements']
    )
    if all_recommendations:
        print(f"  - Top recommendation: {all_recommendations[0]['description']}\n")
    else:
        print(f"  - No recommendations generated\n")

    # Step 5: Polyreactivity Analysis
    print("Step 5: Polyreactivity Analysis")
    # Example antibodies for polyreactivity analysis
    antibodies = {
        'ab1': ('EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSGGGSTYYADSVKGRFTISRDNSKNTLFLQMNSLRAEDTAVYYCARDESGYYYYYYMDVWGQGTTVTVSS',
                'DIQMTQSPSSLSASVGDRVTITCRASQDVSTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSSYPLTFGQGTKVEIK')
    }

    polyreactivity_analyzer = FLAbPolyreactivityAnalyzer()
    polyreactivity_results = polyreactivity_analyzer.analyze_antibodies(antibodies)
    print(f"  - Polyreactivity analysis completed for {len(polyreactivity_results)} antibodies")
    print(f"  - Charge difference for ab1: {polyreactivity_results['ab1']['charge_imbalance']['charge_difference']:.4f}")
    print(f"  - Total hydrophobic patches for ab1: {polyreactivity_results['ab1']['total_hydrophobic_patches']}\n")

    # Step 6: Validation Systems
    print("Step 6: Validation Systems")
    # Initialize validation systems
    systematic_validator = SystematicValidationProtocol()
    drift_detector = ConceptDriftDetector()
    qa_system = SubmissionQualityAssurance()
    prospective_validator = ProspectiveValidationFramework()

    print("  - Validation systems initialized successfully\n")

    print("=== FLAb Framework Test Completed Successfully ===")


def main():
    """
    Main function to run the FLAb framework test.
    """
    test_flab_framework()


if __name__ == '__main__':
    main()
