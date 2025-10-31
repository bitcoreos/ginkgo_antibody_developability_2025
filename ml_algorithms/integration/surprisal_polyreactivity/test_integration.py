#!/usr/bin/env python3
"""
Test script for the surprisal-polyreactivity integration module.
"""

import sys
import os

# Add path to import the integration module
sys.path.append('/a0/bitcore/workspace/ml_algorithms/integration/surprisal_polyreactivity')

try:
    from integration_module import SurprisalPolyreactivityIntegrator
    print("SUCCESS: Integration module imported successfully")
    
    # Example sequences
    vh_sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    vl_sequence = "DIQMTQSPSSLSASVGDRVTITCRASQGIRNYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYGSSPYTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
    
    # Example training sequences for Markov model
    training_sequences = [
        "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS",
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYGMSWVRQAPGKGLEWVSAISSSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDRGGYYAMDYWGQGTMVTVSS",
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGMNWVRQAPGQGLEWMGWISAGSGNTNYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGKGNYYAMDYWGQGTLVTVSS",
        "EVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNISWVRQAPGQGLEWMGWISSSGNTIYYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGKGNYYAMDYWGQGTMVTVSS",
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMSWVRQAPGQGLEWMGWISAGSGNTIYYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGKGNYYAMDYWGQGTLVTVSS"
    ]
    
    # Create integrator
    integrator = SurprisalPolyreactivityIntegrator()
    print("SUCCESS: Integrator created successfully")
    
    # Initialize Markov model
    integrator.initialize_markov_model(training_sequences)
    print("SUCCESS: Markov model initialized successfully")
    
    # Compute comprehensive features
    comprehensive_features = integrator.compute_comprehensive_features(vh_sequence, vl_sequence)
    print("SUCCESS: Comprehensive features computed successfully")
    
    # Check that all expected features are present
    expected_features = ['vh_sequence', 'vl_sequence', 'surprisal_features', 'charge_features', 'clustering_features', 'hydrophobic_features', 'dynamics_features']
    for feature in expected_features:
        if feature in comprehensive_features:
            print(f"SUCCESS: {feature} computed successfully")
        else:
            print(f"ERROR: {feature} missing from comprehensive features")
    
    # Compute integrated risk score
    integrated_risk = integrator.compute_integrated_risk_score(comprehensive_features)
    print("SUCCESS: Integrated risk score computed successfully")
    
    # Check that all expected risk components are present
    expected_risk_components = ['integrated_risk_score', 'burden_q', 'charge_score', 'clustering_score', 'hydrophobic_score', 'dynamics_score', 'weights', 'interpretation', 'scoring_complete']
    for component in expected_risk_components:
        if component in integrated_risk:
            print(f"SUCCESS: {component} computed successfully")
        else:
            print(f"ERROR: {component} missing from integrated risk")
    
    # Generate comprehensive report
    comprehensive_report = integrator.generate_comprehensive_report(vh_sequence, vl_sequence)
    print("SUCCESS: Comprehensive report generated successfully")
    
    # Print final integrated risk score
    print(f"\nFINAL INTEGRATED RISK SCORE: {comprehensive_report['integrated_risk']['integrated_risk_score']:.3f}")
    print(f"INTERPRETATION: {comprehensive_report['integrated_risk']['interpretation']}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
