"""
Markov Models and Surprisal Calculations for Antibody Sequence Analysis
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from itertools import product
import yaml
import math

# Amino acid alphabet
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


class MarkovModel:
    """
    Markov Model for antibody sequence analysis
    """
    
    def __init__(self, order=1, alphabet=AMINO_ACIDS):
        """
        Initialize Markov model
        
        Parameters:
        order (int): Order of the Markov model (1 for first-order, 2 for second-order, etc.)
        alphabet (str): Alphabet of symbols (amino acids)
        """
        self.order = order
        self.alphabet = alphabet
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)
        
    def train(self, sequences):
        """
        Train the Markov model on a set of sequences
        
        Parameters:
        sequences (list): List of sequences to train on
        """
        # Reset model
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)
        
        # Count transitions
        for seq in sequences:
            # Pad sequence with start and end symbols
            padded_seq = '^' * self.order + seq + '$'
            
            # Extract contexts and next symbols
            for i in range(len(padded_seq) - self.order):
                context = padded_seq[i:i+self.order]
                next_symbol = padded_seq[i+self.order]
                
                self.transitions[context][next_symbol] += 1
                self.context_counts[context] += 1
    
    def get_probability(self, context, symbol):
        """
        Get transition probability P(symbol|context)
        
        Parameters:
        context (str): Context sequence
        symbol (str): Next symbol
        
        Returns:
        float: Transition probability
        """
        if context not in self.context_counts or self.context_counts[context] == 0:
            # Return uniform probability for unseen contexts
            return 1.0 / len(self.alphabet)
        
        return self.transitions[context][symbol] / self.context_counts[context]
    
    def get_log_probability(self, context, symbol):
        """
        Get log transition probability log(P(symbol|context))
        
        Parameters:
        context (str): Context sequence
        symbol (str): Next symbol
        
        Returns:
        float: Log transition probability
        """
        prob = self.get_probability(context, symbol)
        return math.log(prob) if prob > 0 else float('-inf')


class SurprisalCalculator:
    """
    Surprisal Calculator for antibody sequence analysis
    """
    
    def __init__(self, markov_model):
        """
        Initialize surprisal calculator
        
        Parameters:
        markov_model (MarkovModel): Trained Markov model
        """
        self.markov_model = markov_model
    
    def calculate_local_surprisal(self, sequence, k=1):
        """
        Calculate local surprisal for a sequence
        Sk(i) = -log p(si..i+k-1)
        
        Parameters:
        sequence (str): Input sequence
        k (int): Length of k-mer to consider
        
        Returns:
        list: List of local surprisal values
        """
        surprisals = []
        padded_seq = '^' * self.markov_model.order + sequence + '$'
        
        for i in range(len(sequence)):
            # Extract context and k-mer
            context = padded_seq[i:i+self.markov_model.order]
            kmer = padded_seq[i+self.markov_model.order:i+self.markov_model.order+k]
            
            # Calculate probability of k-mer given context
            log_prob = 0
            for j in range(len(kmer)):
                if j == 0:
                    log_prob += self.markov_model.get_log_probability(context, kmer[j])
                else:
                    # For longer k-mers, extend context
                    extended_context = context + kmer[:j]
                    log_prob += self.markov_model.get_log_probability(extended_context[-self.markov_model.order:], kmer[j])
            
            # Calculate surprisal
            surprisal = -log_prob if log_prob != float('-inf') else float('inf')
            surprisals.append(surprisal)
        
        return surprisals
    
    def calculate_sequence_surprisal(self, sequence, k=1):
        """
        Calculate overall surprisal for a sequence
        
        Parameters:
        sequence (str): Input sequence
        k (int): Length of k-mer to consider
        
        Returns:
        float: Overall surprisal value
        """
        local_surprisals = self.calculate_local_surprisal(sequence, k)
        return np.mean(local_surprisals) if local_surprisals else 0
    
    def calculate_surprisal_tiers(self, sequences, k=1):
        """
        Calculate surprisal tiers for a set of sequences
        
        Parameters:
        sequences (list): List of sequences
        k (int): Length of k-mer to consider
        
        Returns:
        dict: Dictionary with surprisal tiers
        """
        # Calculate surprisal values for all sequences
        surprisals = [self.calculate_sequence_surprisal(seq, k) for seq in sequences]
        
        # Calculate quantiles for tiering
        quantiles = np.percentile(surprisals, [25, 50, 75])
        
        # Assign tiers
        tiers = []
        for surprisal in surprisals:
            if surprisal <= quantiles[0]:
                tier = 'T0'  # Low risk
            elif surprisal <= quantiles[1]:
                tier = 'T1'  # Medium risk
            elif surprisal <= quantiles[2]:
                tier = 'T2'  # High risk
            else:
                tier = 'T3'  # Very high risk
            tiers.append(tier)
        
        return {
            'surprisals': surprisals,
            'tiers': tiers,
            'quantiles': quantiles.tolist()
        }
    
    def calculate_burden_metrics(self, sequences, k=1):
        """
        Calculate burden metrics for a set of sequences
        
        Parameters:
        sequences (list): List of sequences
        k (int): Length of k-mer to consider
        
        Returns:
        dict: Dictionary with burden metrics
        """
        # Calculate surprisal values for all sequences
        surprisals = [self.calculate_sequence_surprisal(seq, k) for seq in sequences]
        
        # Calculate burden metrics
        burden_q = np.percentile(surprisals, [5, 25, 50, 75, 95])
        s_mean = np.mean(surprisals)
        s_max = np.max(surprisals)
        
        return {
            'burden_q': burden_q.tolist(),
            's_mean': float(s_mean),
            's_max': float(s_max)
        }


def load_config(config_path):
    """
    Load configuration from YAML file
    
    Parameters:
    config_path (str): Path to configuration file
    
    Returns:
    dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """
    Main function for testing
    """
    # Example usage
    sequences = [
        'ACDEFGHIKLMNPQRSTVWY',
        'AAAAAAAAAAAAAAAAAAAA',
        'ACACACACACACACACACAC',
        'DEDEDEDEDEDEDEDEDEDE'
    ]
    
    # Create and train Markov model
    model = MarkovModel(order=1)
    model.train(sequences)
    
    # Create surprisal calculator
    calculator = SurprisalCalculator(model)
    
    # Calculate surprisal for a sequence
    sequence = 'ACDEFGHIKLMNPQRSTVWY'
    surprisal = calculator.calculate_sequence_surprisal(sequence, k=1)
    print(f'Surprisal for sequence {sequence}: {surprisal}')
    
    # Calculate surprisal tiers
    tiers = calculator.calculate_surprisal_tiers(sequences, k=1)
    print(f'Surprisal tiers: {tiers}')
    
    # Calculate burden metrics
    burden = calculator.calculate_burden_metrics(sequences, k=1)
    print(f'Burden metrics: {burden}')


if __name__ == '__main__':
    main()
