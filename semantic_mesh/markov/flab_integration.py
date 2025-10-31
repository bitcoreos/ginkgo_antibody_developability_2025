"""
FLAb Framework Integration for Markov Models and Surprisal Calculations
"""

import sys
import os
import json
import numpy as np
import pandas as pd

# Add paths for FLAb framework and Markov models
sys.path.append('/a0/bitcore/workspace/flab_framework')
sys.path.append('/a0/bitcore/workspace/semantic_mesh/markov')

from markov_models import MarkovModel, SurprisalCalculator


class FLAbMarkovAnalyzer:
    """
    FLAb Framework Integration for Markov Models and Surprisal Analysis
    """
    
    def __init__(self, order=1, kmer_size=1):
        """
        Initialize FLAb Markov Analyzer
        
        Parameters:
        order (int): Order of the Markov model
        kmer_size (int): Size of k-mers for surprisal calculation
        """
        self.order = order
        self.kmer_size = kmer_size
        self.model = None
        self.calculator = None
    
    def train_model(self, sequences):
        """
        Train Markov model on a set of sequences
        
        Parameters:
        sequences (list): List of sequences for training
        
        Returns:
        dict: Training summary
        """
        self.model = MarkovModel(order=self.order)
        self.model.train(sequences)
        self.calculator = SurprisalCalculator(self.model)
        
        return {
            'status': 'success',
            'model_order': self.order,
            'training_sequences': len(sequences)
        }
    
    def analyze_sequence(self, sequence_id, sequence):
        """
        Analyze a single sequence using Markov models and surprisal calculations
        
        Parameters:
        sequence_id (str): Identifier for the sequence
        sequence (str): Amino acid sequence
        
        Returns:
        dict: Analysis results
        """
        if not self.model or not self.calculator:
            raise ValueError("Model not trained. Call train_model first.")
        
        # Calculate surprisal metrics
        local_surprisals = self.calculator.calculate_local_surprisal(sequence, self.kmer_size)
        sequence_surprisal = self.calculator.calculate_sequence_surprisal(sequence, self.kmer_size)
        
        # Calculate statistical features
        surprisal_mean = np.mean(local_surprisals) if local_surprisals else 0
        surprisal_std = np.std(local_surprisals) if local_surprisals else 0
        surprisal_max = np.max(local_surprisals) if local_surprisals else 0
        surprisal_min = np.min(local_surprisals) if local_surprisals else 0
        
        return {
            'sequence_id': sequence_id,
            'sequence_length': len(sequence),
            'surprisal_mean': float(surprisal_mean),
            'surprisal_std': float(surprisal_std),
            'surprisal_max': float(surprisal_max),
            'surprisal_min': float(surprisal_min),
            'sequence_surprisal': float(sequence_surprisal),
            'local_surprisals': [float(s) for s in local_surprisals[:10]]  # First 10 values
        }
    
    def analyze_sequences(self, sequences_dict):
        """
        Analyze multiple sequences
        
        Parameters:
        sequences_dict (dict): Dictionary mapping sequence IDs to sequences
        
        Returns:
        dict: Analysis results for all sequences
        """
        results = {}
        for seq_id, sequence in sequences_dict.items():
            results[seq_id] = self.analyze_sequence(seq_id, sequence)
        
        return results
    
    def get_surprisal_tiers(self, sequences_dict):
        """
        Calculate surprisal tiers for a set of sequences
        
        Parameters:
        sequences_dict (dict): Dictionary mapping sequence IDs to sequences
        
        Returns:
        dict: Surprisal tiers for sequences
        """
        if not self.model or not self.calculator:
            raise ValueError("Model not trained. Call train_model first.")
        
        sequences = list(sequences_dict.values())
        tiers = self.calculator.calculate_surprisal_tiers(sequences, self.kmer_size)
        
        # Map tiers back to sequence IDs
        tier_mapping = {}
        seq_ids = list(sequences_dict.keys())
        for i, seq_id in enumerate(seq_ids):
            tier_mapping[seq_id] = tiers['tiers'][i] if i < len(tiers['tiers']) else 'T0'
        
        return {
            'tiers': tier_mapping,
            'quantiles': tiers['quantiles']
        }
    
    def get_burden_metrics(self, sequences_dict):
        """
        Calculate burden metrics for a set of sequences
        
        Parameters:
        sequences_dict (dict): Dictionary mapping sequence IDs to sequences
        
        Returns:
        dict: Burden metrics
        """
        if not self.model or not self.calculator:
            raise ValueError("Model not trained. Call train_model first.")
        
        sequences = list(sequences_dict.values())
        burden = self.calculator.calculate_burden_metrics(sequences, self.kmer_size)
        
        return burden
    
    def export_features(self, sequences_dict, output_path):
        """
        Export Markov-based features to a CSV file
        
        Parameters:
        sequences_dict (dict): Dictionary mapping sequence IDs to sequences
        output_path (str): Path to output CSV file
        
        Returns:
        dict: Export summary
        """
        # Analyze sequences
        results = self.analyze_sequences(sequences_dict)
        
        # Convert to DataFrame
        df_data = []
        for seq_id, result in results.items():
            df_data.append({
                'sequence_id': seq_id,
                'sequence_length': result['sequence_length'],
                'surprisal_mean': result['surprisal_mean'],
                'surprisal_std': result['surprisal_std'],
                'surprisal_max': result['surprisal_max'],
                'surprisal_min': result['surprisal_min'],
                'sequence_surprisal': result['sequence_surprisal']
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)
        
        return {
            'status': 'success',
            'sequences_processed': len(df_data),
            'output_file': output_path
        }


def main():
    """
    Main function for testing FLAb Markov Analyzer
    """
    print("Testing FLAb Markov Analyzer Integration...")
    
    # Example sequences
    sequences = {
        'seq1': 'ACDEFGHIKLMNPQRSTVWY',
        'seq2': 'AAAAAAAAAAAAAAAAAAAA',
        'seq3': 'ACACACACACACACACACAC',
        'seq4': 'DEDEDEDEDEDEDEDEDEDE'
    }
    
    # Create analyzer
    analyzer = FLAbMarkovAnalyzer(order=1, kmer_size=1)
    
    # Train model
    train_result = analyzer.train_model(list(sequences.values()))
    print(f"Training result: {train_result}")
    
    # Analyze sequences
    results = analyzer.analyze_sequences(sequences)
    print(f"Analysis results for seq1: {results['seq1']}")
    
    # Get surprisal tiers
    tiers = analyzer.get_surprisal_tiers(sequences)
    print(f"Surprisal tiers: {tiers['tiers']}")
    
    # Get burden metrics
    burden = analyzer.get_burden_metrics(sequences)
    print(f"Burden metrics: {burden}")
    
    # Export features
    export_result = analyzer.export_features(sequences, '/a0/bitcore/workspace/semantic_mesh/markov/test_features.csv')
    print(f"Export result: {export_result}")
    
    print("\nFLAb Markov Analyzer test completed successfully!")


if __name__ == '__main__':
    main()
