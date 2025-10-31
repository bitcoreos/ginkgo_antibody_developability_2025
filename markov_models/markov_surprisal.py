"""
Markov Models & Surprisal Calculations Implementation

This module implements local sequence surprisal, Markov models for human repertoire,
surprisal-tiering protocol, risk stratification tiers, and integration into
polyreactivity risk models.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Any
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Amino acid alphabet
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


class MarkovModel:
    """
    Markov model implementation for k-mer background models.
    """
    
    def __init__(self, order: int = 1, pseudo_count: float = 1.0):
        """
        Initialize the Markov model.
        
        Args:
            order (int): Order of the Markov model (1 for first-order, 2 for second-order, etc.)
            pseudo_count (float): Pseudo-count for smoothing
        """
        self.order = order
        self.pseudo_count = pseudo_count
        self.transition_counts = defaultdict(lambda: defaultdict(float))
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.is_trained = False
    
    def train(self, sequences: List[str]):
        """
        Train the Markov model on a set of sequences.
        
        Args:
            sequences (List[str]): List of training sequences
        """
        # Reset counts
        self.transition_counts = defaultdict(lambda: defaultdict(float))
        
        # Count transitions
        for sequence in sequences:
            # Pad sequence with start and end symbols
            padded_sequence = '^' * self.order + sequence + '$'
            
            # Count transitions
            for i in range(len(padded_sequence) - self.order):
                context = padded_sequence[i:i+self.order]
                next_char = padded_sequence[i+self.order]
                self.transition_counts[context][next_char] += 1
        
        # Calculate transition probabilities with smoothing
        for context, next_chars in self.transition_counts.items():
            total_count = sum(next_chars.values()) + len(AMINO_ACIDS) * self.pseudo_count
            for next_char, count in next_chars.items():
                self.transition_probs[context][next_char] = (count + self.pseudo_count) / total_count
            
            # Add pseudo-counts for unseen amino acids
            for aa in AMINO_ACIDS + '$':
                if aa not in self.transition_probs[context]:
                    self.transition_probs[context][aa] = self.pseudo_count / total_count
        
        self.is_trained = True
    
    def get_transition_probability(self, context: str, next_char: str) -> float:
        """
        Get the transition probability for a context and next character.
        
        Args:
            context (str): Context (previous characters)
            next_char (str): Next character
            
        Returns:
            float: Transition probability
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting transition probabilities")
        
        # Pad context if necessary
        if len(context) > self.order:
            context = context[-self.order:]
        elif len(context) < self.order:
            context = '^' * (self.order - len(context)) + context
        
        return self.transition_probs[context].get(next_char, self.pseudo_count / (len(AMINO_ACIDS) + 1))
    
    def generate_sequence(self, length: int) -> str:
        """
        Generate a sequence using the trained Markov model.
        
        Args:
            length (int): Length of sequence to generate
            
        Returns:
            str: Generated sequence
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating sequences")
        
        # Start with order symbols
        sequence = '^' * self.order
        
        # Generate sequence
        for _ in range(length):
            context = sequence[-self.order:]
            next_chars = list(self.transition_probs[context].keys())
            probabilities = list(self.transition_probs[context].values())
            
            # Normalize probabilities
            probabilities = np.array(probabilities)
            probabilities = probabilities / np.sum(probabilities)
            
            # Sample next character
            next_char = np.random.choice(next_chars, p=probabilities)
            
            # Stop if end symbol
            if next_char == '$':
                break
            
            sequence += next_char
        
        # Remove padding
        sequence = sequence[self.order:]
        
        return sequence


class SurprisalCalculator:
    """
    Local sequence surprisal calculation using k-mer background models.
    """
    
    def __init__(self, markov_model: MarkovModel):
        """
        Initialize the surprisal calculator.
        
        Args:
            markov_model (MarkovModel): Trained Markov model for background probabilities
        """
        self.markov_model = markov_model
    
    def calculate_local_surprisal(self, sequence: str, k: int = 3) -> List[float]:
        """
        Calculate local sequence surprisal (Sk(i) = -log p(si..i+k-1)).
        
        Args:
            sequence (str): Input sequence
            k (int): Length of k-mers for surprisal calculation
            
        Returns:
            List[float]: List of surprisal values for each position
        """
        surprisal_values = []
        
        # Calculate surprisal for each k-mer
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            context = kmer[:-1]  # All but last character
            next_char = kmer[-1]  # Last character
            
            # Get transition probability
            prob = self.markov_model.get_transition_probability(context, next_char)
            
            # Calculate surprisal
            surprisal = -np.log(prob)
            surprisal_values.append(surprisal)
        
        return surprisal_values
    
    def calculate_sliding_surprisal(self, sequence: str, k: int = 3, window_size: int = 5) -> List[float]:
        """
        Calculate sliding window surprisal.
        
        Args:
            sequence (str): Input sequence
            k (int): Length of k-mers for surprisal calculation
            window_size (int): Size of sliding window
            
        Returns:
            List[float]: List of average surprisal values for each window
        """
        local_surprisal = self.calculate_local_surprisal(sequence, k)
        sliding_surprisal = []
        
        # Calculate sliding window averages
        for i in range(len(local_surprisal) - window_size + 1):
            window_avg = np.mean(local_surprisal[i:i+window_size])
            sliding_surprisal.append(window_avg)
        
        return sliding_surprisal


class SurprisalTiering:
    """
    Surprisal-tiering protocol with burden metrics.
    """
    
    def __init__(self):
        """
        Initialize the surprisal tiering protocol.
        """
        pass
    
    def calculate_burden_metrics(self, surprisal_values: List[float]) -> Dict[str, float]:
        """
        Calculate burden metrics (Burden_q, S-mean, S-max).
        
        Args:
            surprisal_values (List[float]): List of surprisal values
            
        Returns:
            Dict[str, float]: Burden metrics
        """
        if not surprisal_values:
            return {
                'burden_q': 0.0,
                's_mean': 0.0,
                's_max': 0.0,
                's_std': 0.0
            }
        
        surprisal_array = np.array(surprisal_values)
        
        # Calculate metrics
        s_mean = np.mean(surprisal_array)
        s_max = np.max(surprisal_array)
        s_std = np.std(surprisal_array)
        
        # Burden_q (quantile-based metric)
        # Higher values indicate more high-surprisal regions
        burden_q = np.percentile(surprisal_array, 90)  # 90th percentile
        
        return {
            'burden_q': float(burden_q),
            's_mean': float(s_mean),
            's_max': float(s_max),
            's_std': float(s_std)
        }
    
    def stratify_risk_tiers(self, surprisal_values: List[float]) -> Dict[str, Union[int, str, Dict]]:
        """
        Stratify risk tiers (T0-T3) based on surprisal quantiles.
        
        Args:
            surprisal_values (List[float]): List of surprisal values
            
        Returns:
            Dict[str, Union[int, str, Dict]]: Risk stratification results
        """
        if not surprisal_values:
            return {
                'tier': 0,
                'tier_label': 'T0',
                'risk_level': 'low',
                'recommendation': 'No surprisal data available',
                'thresholds': {}
            }
        
        surprisal_array = np.array(surprisal_values)
        
        # Define quantile thresholds for tiers
        q25 = np.percentile(surprisal_array, 25)
        q50 = np.percentile(surprisal_array, 50)
        q75 = np.percentile(surprisal_array, 75)
        
        # Calculate mean surprisal for tier assignment
        mean_surprisal = np.mean(surprisal_array)
        
        # Assign tier based on mean surprisal
        if mean_surprisal <= q25:
            tier = 0
            tier_label = 'T0'
            risk_level = 'low'
            recommendation = 'Low surprisal burden - good developability profile'
        elif mean_surprisal <= q50:
            tier = 1
            tier_label = 'T1'
            risk_level = 'low-medium'
            recommendation = 'Moderate surprisal burden - acceptable developability profile'
        elif mean_surprisal <= q75:
            tier = 2
            tier_label = 'T2'
            risk_level = 'medium-high'
            recommendation = 'High surprisal burden - consider optimization'
        else:
            tier = 3
            tier_label = 'T3'
            risk_level = 'high'
            recommendation = 'Very high surprisal burden - significant optimization recommended'
        
        return {
            'tier': tier,
            'tier_label': tier_label,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'thresholds': {
                'q25': float(q25),
                'q50': float(q50),
                'q75': float(q75),
                'mean_surprisal': float(mean_surprisal)
            }
        }


class HumanRepertoireMarkov:
    """
    IGHV/IGKV human repertoire Markov models (order 1-3).
    """
    
    def __init__(self):
        """
        Initialize the human repertoire Markov models.
        """
        self.models = {}
        self.is_trained = False
    
    def train_models(self, ighv_sequences: List[str], igkv_sequences: List[str]):
        """
        Train Markov models for IGHV and IGKV human repertoires.
        
        Args:
            ighv_sequences (List[str]): List of IGHV sequences
            igkv_sequences (List[str]): List of IGKV sequences
        """
        # Train models of different orders for IGHV
        for order in [1, 2, 3]:
            model = MarkovModel(order=order)
            model.train(ighv_sequences)
            self.models[f'ighv_order_{order}'] = model
        
        # Train models of different orders for IGKV
        for order in [1, 2, 3]:
            model = MarkovModel(order=order)
            model.train(igkv_sequences)
            self.models[f'igkv_order_{order}'] = model
        
        self.is_trained = True
    
    def get_model(self, chain_type: str, order: int) -> MarkovModel:
        """
        Get a trained Markov model.
        
        Args:
            chain_type (str): Chain type ('ighv' or 'igkv')
            order (int): Model order (1, 2, or 3)
            
        Returns:
            MarkovModel: Trained Markov model
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before retrieval")
        
        model_key = f'{chain_type}_order_{order}'
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not found")
        
        return self.models[model_key]


class SurprisalPolyreactivityIntegration:
    """
    Integration of surprisal tiers into polyreactivity risk models.
    """
    
    def __init__(self):
        """
        Initialize the surprisal-polyreactivity integration.
        """
        pass
    
    def integrate_surprisal_risk(self, surprisal_tiering: Dict, 
                               base_polyreactivity_risk: float) -> Dict[str, Union[float, str]]:
        """
        Integrate surprisal tiers into polyreactivity risk models.
        
        Args:
            surprisal_tiering (Dict): Surprisal tiering results
            base_polyreactivity_risk (float): Base polyreactivity risk score
            
        Returns:
            Dict[str, Union[float, str]]: Integrated risk assessment
        """
        # Get tier information
        tier = surprisal_tiering.get('tier', 0)
        tier_label = surprisal_tiering.get('tier_label', 'T0')
        
        # Adjust polyreactivity risk based on surprisal tier
        # Higher surprisal tiers increase polyreactivity risk
        tier_multiplier = 1.0 + (tier * 0.2)  # 20% increase per tier
        adjusted_risk = base_polyreactivity_risk * tier_multiplier
        
        # Cap at 1.0
        adjusted_risk = min(adjusted_risk, 1.0)
        
        # Determine risk level
        if adjusted_risk < 0.3:
            risk_level = 'low'
            recommendation = 'Low integrated risk - suitable for development'
        elif adjusted_risk < 0.6:
            risk_level = 'medium'
            recommendation = 'Moderate integrated risk - consider optimization'
        else:
            risk_level = 'high'
            recommendation = 'High integrated risk - significant optimization recommended'
        
        return {
            'adjusted_polyreactivity_risk': float(adjusted_risk),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'surprisal_tier': tier_label,
            'tier_multiplier': float(tier_multiplier),
            'base_risk': float(base_polyreactivity_risk)
        }


def main():
    """
    Example usage of the Markov models and surprisal calculations implementation.
    """
    # Example sequences for training
    ighv_sequences = [
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGMNWVRQAPGQGLEWMGWISAGSGSTYYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGGTFAYYAMDYWGQGTLVTVSS",
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVARIYYSGSTNYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS",
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGMNWVRQAPGQGLEWMGWISAGSGSTYYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGGTFAYYAMDYWGQGTLVTVSS"
    ]
    
    igkv_sequences = [
        "DIQMTQSPSSLSASVGDRVTITCRASQSVSSSYLAWYQQKPGKAPKLLIYDASNRATGIPDRFSGSGSGTDFTLTISSLQPEDFATYYCQQRSNWPRTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC",
        "DIVMTQSPDSLAVSLGERATINCKSSQSVLYHSNKNYLAWYQQKPGQSPKLLIYWASTRESGVPDRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSSYPLTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC",
        "DIQMTQSPSSLSASVGDRVTITCRASQSVSSSYLAWYQQKPGKAPKLLIYDASNRATGIPDRFSGSGSGTDFTLTISSLQPEDFATYYCQQRSNWPRTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
    ]
    
    # Train human repertoire Markov models
    human_markov = HumanRepertoireMarkov()
    human_markov.train_models(ighv_sequences, igkv_sequences)
    
    print("Human Repertoire Markov Models Training Results:")
    print(f"  Trained models: {list(human_markov.models.keys())}")
    
    # Get a specific model
    ighv_model = human_markov.get_model('ighv', 2)
    
    # Example sequence for surprisal calculation
    example_sequence = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGMNWVRQAPGQGLEWMGWISAGSGSTYYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGGTFAYYAMDYWGQGTLVTVSS"
    
    # Calculate surprisal
    surprisal_calc = SurprisalCalculator(ighv_model)
    local_surprisal = surprisal_calc.calculate_local_surprisal(example_sequence, k=3)
    sliding_surprisal = surprisal_calc.calculate_sliding_surprisal(example_sequence, k=3, window_size=5)
    
    print("\nSurprisal Calculation Results:")
    print(f"  Local surprisal (first 10 values): {local_surprisal[:10]}")
    print(f"  Sliding surprisal (first 10 values): {sliding_surprisal[:10]}")
    print(f"  Mean local surprisal: {np.mean(local_surprisal):.4f}")
    print(f"  Max local surprisal: {np.max(local_surprisal):.4f}")
    
    # Calculate burden metrics
    surprisal_tiering = SurprisalTiering()
    burden_metrics = surprisal_tiering.calculate_burden_metrics(local_surprisal)
    
    print("\nBurden Metrics:")
    print(f"  Burden_q (90th percentile): {burden_metrics['burden_q']:.4f}")
    print(f"  S-mean: {burden_metrics['s_mean']:.4f}")
    print(f"  S-max: {burden_metrics['s_max']:.4f}")
    print(f"  S-std: {burden_metrics['s_std']:.4f}")
    
    # Stratify risk tiers
    risk_tiers = surprisal_tiering.stratify_risk_tiers(local_surprisal)
    
    print("\nRisk Stratification:")
    print(f"  Tier: {risk_tiers['tier_label']} (Tier {risk_tiers['tier']})")
    print(f"  Risk Level: {risk_tiers['risk_level']}")
    print(f"  Recommendation: {risk_tiers['recommendation']}")
    
    # Integrate surprisal into polyreactivity risk model
    integration = SurprisalPolyreactivityIntegration()
    base_polyreactivity_risk = 0.4  # Example base risk
    integrated_risk = integration.integrate_surprisal_risk(risk_tiers, base_polyreactivity_risk)
    
    print("\nIntegrated Risk Assessment:")
    print(f"  Base Polyreactivity Risk: {integrated_risk['base_risk']:.4f}")
    print(f"  Surprisal Tier: {integrated_risk['surprisal_tier']}")
    print(f"  Tier Multiplier: {integrated_risk['tier_multiplier']:.4f}")
    print(f"  Adjusted Risk: {integrated_risk['adjusted_polyreactivity_risk']:.4f}")
    print(f"  Risk Level: {integrated_risk['risk_level']}")
    print(f"  Recommendation: {integrated_risk['recommendation']}")


if __name__ == "__main__":
    main()
