"""
Comprehensive test script for FLAb Framework with Heavy-Light Coupling Integration

This script demonstrates the complete workflow of the FLAb framework with heavy-light coupling analysis.
"""

import sys
import os

# Add paths for polyreactivity analysis and validation systems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, '/a0/bitcore/workspace/polyreactivity_analysis')
# sys.path.insert(0, '/a0/bitcore/workspace/validation_systems')
# sys.path.insert(0, '/a0/bitcore/workspace/flab_framework/heavy_light_coupling')

# Import all FLAb modules directly from their Python files
from fragment_analyzer.fragment_analyzer import FragmentAnalyzer
from fragment_database import FragmentDatabase
from developability_predictor import DevelopabilityPredictor
from optimization_recommender import OptimizationRecommender

# Import polyreactivity analysis
from flab_integration import FLAbPolyreactivityAnalyzer

# Import validation systems
from validation_framework import SystematicValidationProtocol, ConceptDriftDetector, SubmissionQualityAssurance, ProspectiveValidationFramework

# Import heavy-light coupling analysis
from flab_heavy_light_analyzer import FLAbHeavyLightAnalyzer

def test_flab_framework_with_heavy_light_coupling():
    """
    Test the complete FLAb framework workflow with heavy-light coupling analysis.
    """
    print("=== FLAb Framework Comprehensive Test with Heavy-Light Coupling ===\n")

    # Test sequences
    vh_sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    vl_sequence = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLAWYQQKPGKAPKLLIYDASNLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSNHWPTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
    heavy_chain_sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSREEMTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK"
    light_chain_sequence = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLAWYQQKPGKAPKLLIYDASNLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSNHWPTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
    fragment_id = "test_antibody_001"

    print(f"Analyzing antibody: {fragment_id}")
    print(f"VH sequence length: {len(vh_sequence)}")
    print(f"VL sequence length: {len(vl_sequence)}")
    print(f"Heavy chain sequence length: {len(heavy_chain_sequence)}")
    print(f"Light chain sequence length: {len(light_chain_sequence)}\n")

    # Step 1: Fragment Analysis
    print("Step 1: Fragment Analysis")
    analyzer = FragmentAnalyzer()
    vh_analysis = analyzer.analyze_sequence(vh_sequence)
    vl_analysis = analyzer.analyze_sequence(vl_sequence)
    vh_phys_props = analyzer.calculate_physicochemical_properties(vh_sequence)
    vl_phys_props = analyzer.calculate_physicochemical_properties(vl_sequence)
    vh_stability = analyzer.assess_stability(vh_sequence)
    vl_stability = analyzer.assess_stability(vl_sequence)

    # Combine VH analysis results
    vh_full_analysis = {
        'sequence': vh_sequence,
        'composition': vh_analysis['composition'],
        'complexity': vh_analysis['complexity'],
        'physicochemical_properties': vh_phys_props,
        'stability': vh_stability
    }

    # Combine VL analysis results
    vl_full_analysis = {
        'sequence': vl_sequence,
        'composition': vl_analysis['composition'],
        'complexity': vl_analysis['complexity'],
        'physicochemical_properties': vl_phys_props,
        'stability': vl_stability
    }

    print(f"  - VH amino acid composition: {vh_analysis['composition']['total_amino_acids']} residues")
    print(f"  - VH sequence complexity: {vh_analysis['complexity']['overall_complexity']:.4f}")
    print(f"  - VH isoelectric point: {vh_phys_props['isoelectric_point']:.2f}")
    print(f"  - VH hydrophobicity: {vh_phys_props['hydrophobicity']:.4f}")
    print(f"  - VH aggregation propensity: {vh_stability['aggregation_propensity']:.4f}")
    print(f"  - VL amino acid composition: {vl_analysis['composition']['total_amino_acids']} residues")
    print(f"  - VL sequence complexity: {vl_analysis['complexity']['overall_complexity']:.4f}")
    print(f"  - VL isoelectric point: {vl_phys_props['isoelectric_point']:.2f}")
    print(f"  - VL hydrophobicity: {vl_phys_props['hydrophobicity']:.4f}")
    print(f"  - VL aggregation propensity: {vl_stability['aggregation_propensity']:.4f}\n")

    # Step 2: Fragment Database Storage
    print("Step 2: Fragment Database Storage")
    db = FragmentDatabase('test_fragment_database.json')
    db.add_fragment(f"{fragment_id}_vh", vh_full_analysis)
    db.add_fragment(f"{fragment_id}_vl", vl_full_analysis)
    stored_vh_fragment = db.get_fragment(f"{fragment_id}_vh")
    stored_vl_fragment = db.get_fragment(f"{fragment_id}_vl")
    print(f"  - VH fragment stored successfully: {stored_vh_fragment is not None}")
    print(f"  - VL fragment stored successfully: {stored_vl_fragment is not None}\n")

    # Step 3: Developability Prediction
    print("Step 3: Developability Prediction")
    predictor = DevelopabilityPredictor()
    vh_predictions = predictor.predict_developability(vh_full_analysis)
    vl_predictions = predictor.predict_developability(vl_full_analysis)
    print(f"  - VH solubility prediction: {vh_predictions['solubility']:.4f}")
    print(f"  - VH expression level prediction: {vh_predictions['expression']:.4f}")
    print(f"  - VH aggregation propensity prediction: {vh_predictions['aggregation']:.4f}")
    print(f"  - VH immunogenicity prediction: {vh_predictions['immunogenicity']:.4f}")
    print(f"  - VH overall developability score: {vh_predictions['overall_score']:.4f}")
    print(f"  - VL solubility prediction: {vl_predictions['solubility']:.4f}")
    print(f"  - VL expression level prediction: {vl_predictions['expression']:.4f}")
    print(f"  - VL aggregation propensity prediction: {vl_predictions['aggregation']:.4f}")
    print(f"  - VL immunogenicity prediction: {vl_predictions['immunogenicity']:.4f}")
    print(f"  - VL overall developability score: {vl_predictions['overall_score']:.4f}\n")

    # Step 4: Optimization Recommendations
    print("Step 4: Optimization Recommendations")
    recommender = OptimizationRecommender()
    vh_recommendations = recommender.generate_recommendations(vh_full_analysis, vh_predictions)
    vl_recommendations = recommender.generate_recommendations(vl_full_analysis, vl_predictions)
    print(f"  - VH recommendations priority: {vh_recommendations['priority']}")
    print(f"  - VH sequence modifications: {len(vh_recommendations['sequence_modifications'])}")
    print(f"  - VH structural strategies: {len(vh_recommendations['structural_strategies'])}")
    print(f"  - VH design improvements: {len(vh_recommendations['design_improvements'])}")
    print(f"  - VL recommendations priority: {vl_recommendations['priority']}")
    print(f"  - VL sequence modifications: {len(vl_recommendations['sequence_modifications'])}")
    print(f"  - VL structural strategies: {len(vl_recommendations['structural_strategies'])}")
    print(f"  - VL design improvements: {len(vl_recommendations['design_improvements'])}")
    
    # Print first recommendation if any exist
    all_vh_recommendations = (
        vh_recommendations['sequence_modifications'] +
        vh_recommendations['structural_strategies'] +
        vh_recommendations['design_improvements']
    )
    if all_vh_recommendations:
        print(f"  - Top VH recommendation: {all_vh_recommendations[0]['description']}")
    
    all_vl_recommendations = (
        vl_recommendations['sequence_modifications'] +
        vl_recommendations['structural_strategies'] +
        vl_recommendations['design_improvements']
    )
    if all_vl_recommendations:
        print(f"  - Top VL recommendation: {all_vl_recommendations[0]['description']}")
    print()

    # Step 5: Polyreactivity Analysis
    print("Step 5: Polyreactivity Analysis")
    # Example antibodies for polyreactivity analysis
    antibodies = {
        'ab1': (vh_sequence, vl_sequence)
    }

    polyreactivity_analyzer = FLAbPolyreactivityAnalyzer()
    polyreactivity_results = polyreactivity_analyzer.analyze_antibodies(antibodies)
    print(f"  - Polyreactivity analysis completed for {len(polyreactivity_results)} antibodies")
    print(f"  - Charge difference for ab1: {polyreactivity_results['ab1']['charge_imbalance']['charge_difference']:.4f}")
    print(f"  - Total hydrophobic patches for ab1: {polyreactivity_results['ab1']['total_hydrophobic_patches']}")
    # Note: The current implementation doesn't return aromatic patches or total polyreactivity score
    print(f"  - VH hydrophobic patch count for ab1: {polyreactivity_results['ab1']['vh_hydrophobic_patches']['hydrophobic_patch_count']}")
    print(f"  - VL hydrophobic patch count for ab1: {polyreactivity_results['ab1']['vl_hydrophobic_patches']['hydrophobic_patch_count']}")
    print()

    # Step 6: Heavy-Light Coupling Analysis
    print("Step 6: Heavy-Light Coupling Analysis")
    heavy_light_analyzer = FLAbHeavyLightAnalyzer()
    coupling_report = heavy_light_analyzer.analyze_antibody(
        vh_sequence, vl_sequence, heavy_chain_sequence, light_chain_sequence, 'IgG1'
    )
    pairing_score = heavy_light_analyzer.get_pairing_score(vh_sequence, vl_sequence)
    isotype_score = heavy_light_analyzer.get_isotype_score(heavy_chain_sequence, light_chain_sequence, 'IgG1')
    overall_compatibility = heavy_light_analyzer.get_overall_compatibility(
        vh_sequence, vl_sequence, heavy_chain_sequence, light_chain_sequence, 'IgG1'
    )
    print(f"  - VH gene family: {coupling_report['pairing_analysis']['vh_gene']}")
    print(f"  - VL gene family: {coupling_report['pairing_analysis']['vl_gene']}")
    print(f"  - Pairing score: {pairing_score:.4f}")
    print(f"  - Isotype score: {isotype_score:.4f}")
    print(f"  - Overall compatibility: {overall_compatibility:.4f}")
    print(f"  - Charge compatibility: {coupling_report['pairing_analysis']['charge_compatibility']:.4f}")
    print(f"  - Hydrophobicity compatibility: {coupling_report['pairing_analysis']['hydrophobicity_compatibility']:.4f}")
    print(f"  - Length compatibility: {coupling_report['pairing_analysis']['length_compatibility']:.4f}\n")

    # Step 7: Validation Systems
    print("Step 7: Validation Systems")
    # Initialize validation systems
    systematic_validator = SystematicValidationProtocol()
    drift_detector = ConceptDriftDetector()
    qa_system = SubmissionQualityAssurance()
    prospective_validator = ProspectiveValidationFramework()

    print("  - Validation systems initialized successfully\n")

    print("=== FLAb Framework Test with Heavy-Light Coupling Completed Successfully ===")


def main():
    """
    Main function to run the FLAb framework test with heavy-light coupling.
    """
    test_flab_framework_with_heavy_light_coupling()


if __name__ == '__main__':
    main()
