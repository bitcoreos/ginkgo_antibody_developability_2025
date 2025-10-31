"""
Test script for Markov models and surprisal calculations
"""

import sys
import os
sys.path.append('/a0/bitcore/workspace/semantic_mesh/markov')

from markov_models import MarkovModel, SurprisalCalculator


def test_markov_model():
    """
    Test Markov model functionality
    """
    print("Testing Markov Model...")
    
    # Example sequences
    sequences = [
        'ACDEFGHIKLMNPQRSTVWY',
        'AAAAAAAAAAAAAAAAAAAA',
        'ACACACACACACACACACAC',
        'DEDEDEDEDEDEDEDEDEDE'
    ]
    
    # Create and train Markov model
    model = MarkovModel(order=1)
    model.train(sequences)
    
    # Test probability calculation
    prob = model.get_probability('A', 'C')
    print(f'P(C|A) = {prob}')
    
    # Test log probability calculation
    log_prob = model.get_log_probability('A', 'C')
    print(f'log(P(C|A)) = {log_prob}')
    
    return model


def test_surprisal_calculator(model):
    """
    Test surprisal calculator functionality
    """
    print("\nTesting Surprisal Calculator...")
    
    # Create surprisal calculator
    calculator = SurprisalCalculator(model)
    
    # Test local surprisal calculation
    sequence = 'ACDEFGHIKLMNPQRSTVWY'
    local_surprisals = calculator.calculate_local_surprisal(sequence, k=1)
    print(f'Local surprisals for {sequence}: {local_surprisals[:5]}...')  # Show first 5 values
    
    # Test sequence surprisal calculation
    sequence_surprisal = calculator.calculate_sequence_surprisal(sequence, k=1)
    print(f'Sequence surprisal for {sequence}: {sequence_surprisal}')
    
    # Test surprisal tiers
    sequences = [
        'ACDEFGHIKLMNPQRSTVWY',
        'AAAAAAAAAAAAAAAAAAAA',
        'ACACACACACACACACACAC',
        'DEDEDEDEDEDEDEDEDEDE'
    ]
    
    tiers = calculator.calculate_surprisal_tiers(sequences, k=1)
    print(f'Surprisal tiers: {tiers["tiers"]}')
    
    # Test burden metrics
    burden = calculator.calculate_burden_metrics(sequences, k=1)
    print(f'Burden metrics: S-mean = {burden["s_mean"]}, S-max = {burden["s_max"]}')


def main():
    """
    Main test function
    """
    print("Running Markov Model and Surprisal Calculator Tests\n")
    
    # Test Markov model
    model = test_markov_model()
    
    # Test surprisal calculator
    test_surprisal_calculator(model)
    
    print("\nAll tests completed successfully!")


if __name__ == '__main__':
    main()
