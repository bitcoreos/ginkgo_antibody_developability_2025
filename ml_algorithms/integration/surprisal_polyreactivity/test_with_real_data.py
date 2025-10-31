#!/usr/bin/env python3
"""
Test script for the surprisal-polyreactivity integration module using real GDPa1 dataset sequences.
"""

import sys
import os
import pandas as pd

# Add paths to import the required modules
sys.path.append('/a0/bitcore/workspace/ml_algorithms/integration/surprisal_polyreactivity')
sys.path.append('/a0/bitcore/workspace/research/semantic_mesh/markov/src')
sys.path.append('/a0/bitcore/workspace/research/polyreactivity/src')

try:
    from integration_module import SurprisalPolyreactivityIntegrator
    print("SUCCESS: Integration module imported successfully")
    
    # Load GDPa1 dataset
    df = pd.read_csv('/a0/bitcore/workspace/data/processed/GDPa1_v1.2_sequences_processed.csv')
    print(f"SUCCESS: GDPa1 dataset loaded with {len(df)} entries")
    
    # Extract sample sequences
    sample_data = df[['antibody_id', 'vh_protein_sequence', 'vl_protein_sequence']].head(3)
    print("Sample sequences:")
    for idx, row in sample_data.iterrows():
        print(f"  {row['antibody_id']}: VH length={len(row['vh_protein_sequence'])}, VL length={len(row['vl_protein_sequence'])}")
    
    # Extract training sequences for Markov model (using a larger sample)
    training_sequences = (df['vh_protein_sequence'].head(20).tolist() + 
                          df['vl_protein_sequence'].head(20).tolist())
    print(f"SUCCESS: Extracted {len(training_sequences)} training sequences for Markov model")
    
    # Create integrator
    integrator = SurprisalPolyreactivityIntegrator()
    print("SUCCESS: Integrator created successfully")
    
    # Initialize Markov model
    integrator.initialize_markov_model(training_sequences)
    print("SUCCESS: Markov model initialized successfully")
    
    # Test with sample sequences
    print("\nTesting integration module with real GDPa1 sequences:")
    print("=" * 60)
    
    for idx, row in sample_data.iterrows():
        antibody_id = row['antibody_id']
        vh_sequence = row['vh_protein_sequence']
        vl_sequence = row['vl_protein_sequence']
        
        print(f"\nAnalyzing {antibody_id}:")
        
        # Generate comprehensive report
        comprehensive_report = integrator.generate_comprehensive_report(vh_sequence, vl_sequence)
        
        # Print results
        integrated_risk = comprehensive_report['integrated_risk']
        print(f"  Integrated Risk Score: {integrated_risk['integrated_risk_score']:.3f}")
        print(f"  Interpretation: {integrated_risk['interpretation']}")
        
        print("  Component Scores:")
        print(f"    Burden Q (Surprisal): {integrated_risk['burden_q']:.3f}")
        print(f"    Charge Imbalance: {integrated_risk['charge_score']:.3f}")
        print(f"    Clustering Risk: {integrated_risk['clustering_score']:.3f}")
        print(f"    Hydrophobic Binding Potential: {integrated_risk['hydrophobic_score']:.3f}")
        print(f"    Dynamics Risk: {integrated_risk['dynamics_score']:.3f}")
    
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
