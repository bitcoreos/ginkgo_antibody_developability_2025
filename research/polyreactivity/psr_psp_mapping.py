"""
PSR/PSP Assay Mapping and Decision Rules Implementation

This module implements comprehensive mapping between sequence features
and polyreactivity/self-association (PSR/PSP) assay results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union


class PSRPSPEvaluator:
    """
    A class for evaluating polyreactivity/self-association risk based on
    sequence features and assay mapping.
    """
    
    def __init__(self):
        """
        Initialize the PSR/PSP evaluator with decision rules.
        """
        # Decision thresholds based on literature and empirical data
        self.thresholds = {
            'charge_imbalance': 2.0,  # Absolute charge difference
            'hydrophobic_patch_density': 0.1,  # Patches per residue
            'paratope_entropy': 3.0,  # Bits of entropy
            'negative_charge_fraction': 0.3,  # Fraction of negative charges
            'positive_charge_fraction': 0.3,  # Fraction of positive charges
            'aromatic_fraction': 0.2,  # Fraction of aromatic residues
        }
        
        # Risk weights for different features
        self.weights = {
            'charge_imbalance': 0.25,
            'hydrophobic_patch_density': 0.20,
            'paratope_entropy': 0.20,
            'charge_balance': 0.15,
            'surface_exposed_hydrophobicity': 0.10,
            'aromatic_content': 0.10
        }
    
    def evaluate_charge_risk(self, vh_net_charge: float, vl_net_charge: float, 
                           charge_imbalance: float) -> Dict[str, float]:
        """
        Evaluate polyreactivity risk based on charge features.
        
        Args:
            vh_net_charge (float): Net charge of VH domain
            vl_net_charge (float): Net charge of VL domain
            charge_imbalance (float): Absolute charge imbalance
            
        Returns:
            Dict[str, float]: Charge-based risk metrics
        """
        # Charge imbalance risk
        imbalance_risk = min(charge_imbalance / self.thresholds['charge_imbalance'], 1.0)
        
        # Overall charge risk (considering both imbalance and absolute charges)
        abs_charge_risk = min((abs(vh_net_charge) + abs(vl_net_charge)) / 4.0, 1.0)
        
        # Combined charge risk
        charge_risk = 0.7 * imbalance_risk + 0.3 * abs_charge_risk
        
        return {
            'charge_imbalance_risk': imbalance_risk,
            'absolute_charge_risk': abs_charge_risk,
            'combined_charge_risk': charge_risk
        }
    
    def evaluate_hydrophobicity_risk(self, hydrophobic_patch_density: float, 
                                   surface_exposed_hydrophobicity: float = 0.0) -> Dict[str, float]:
        """
        Evaluate polyreactivity risk based on hydrophobicity features.
        
        Args:
            hydrophobic_patch_density (float): Density of hydrophobic patches
            surface_exposed_hydrophobicity (float): Surface-exposed hydrophobicity
            
        Returns:
            Dict[str, float]: Hydrophobicity-based risk metrics
        """
        # Hydrophobic patch risk
        patch_risk = min(hydrophobic_patch_density / self.thresholds['hydrophobic_patch_density'], 1.0)
        
        # Surface exposure risk
        exposure_risk = min(surface_exposed_hydrophobicity / 0.3, 1.0) if surface_exposed_hydrophobicity > 0 else 0.0
        
        # Combined hydrophobicity risk
        hydro_risk = 0.7 * patch_risk + 0.3 * exposure_risk
        
        return {
            'hydrophobic_patch_risk': patch_risk,
            'surface_exposure_risk': exposure_risk,
            'combined_hydrophobicity_risk': hydro_risk
        }
    
    def evaluate_paratope_risk(self, paratope_entropy: float, 
                             aromatic_fraction: float) -> Dict[str, float]:
        """
        Evaluate polyreactivity risk based on paratope features.
        
        Args:
            paratope_entropy (float): Paratope entropy
            aromatic_fraction (float): Fraction of aromatic residues in paratope
            
        Returns:
            Dict[str, float]: Paratope-based risk metrics
        """
        # Paratope entropy risk
        entropy_risk = min(paratope_entropy / self.thresholds['paratope_entropy'], 1.0)
        
        # Aromatic content risk
        aromatic_risk = min(aromatic_fraction / self.thresholds['aromatic_fraction'], 1.0)
        
        # Combined paratope risk
        paratope_risk = 0.6 * entropy_risk + 0.4 * aromatic_risk
        
        return {
            'paratope_entropy_risk': entropy_risk,
            'aromatic_content_risk': aromatic_risk,
            'combined_paratope_risk': paratope_risk
        }
    
    def evaluate_comprehensive_risk(self, charge_features: Dict[str, float],
                                  hydrophobicity_features: Dict[str, float],
                                  paratope_features: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate comprehensive polyreactivity/self-association risk.
        
        Args:
            charge_features (Dict[str, float]): Charge-based features
            hydrophobicity_features (Dict[str, float]): Hydrophobicity-based features
            paratope_features (Dict[str, float]): Paratope-based features
            
        Returns:
            Dict[str, float]: Comprehensive risk assessment
        """
        # Evaluate individual risks
        charge_risk = self.evaluate_charge_risk(
            charge_features.get('vh_net_charge', 0.0),
            charge_features.get('vl_net_charge', 0.0),
            charge_features.get('charge_imbalance', 0.0)
        )
        
        hydro_risk = self.evaluate_hydrophobicity_risk(
            hydrophobicity_features.get('patch_density', 0.0),
            hydrophobicity_features.get('surface_exposed_hydrophobicity', 0.0)
        )
        
        paratope_risk = self.evaluate_paratope_risk(
            paratope_features.get('paratope_entropy', 0.0),
            paratope_features.get('aromatic_fraction', 0.0)
        )
        
        # Weighted combination of all risks
        comprehensive_risk = (
            self.weights['charge_imbalance'] * charge_risk['combined_charge_risk'] +
            self.weights['hydrophobic_patch_density'] * hydro_risk['combined_hydrophobicity_risk'] +
            self.weights['paratope_entropy'] * paratope_risk['combined_paratope_risk'] +
            self.weights['charge_balance'] * charge_risk['absolute_charge_risk'] +
            self.weights['surface_exposed_hydrophobicity'] * hydro_risk['surface_exposure_risk'] +
            self.weights['aromatic_content'] * paratope_risk['aromatic_content_risk']
        )
        
        # Risk stratification
        if comprehensive_risk < 0.3:
            risk_level = 'Low'
            recommendation = 'Antibody likely to have good developability'
        elif comprehensive_risk < 0.6:
            risk_level = 'Moderate'
            recommendation = 'Consider optimization to reduce polyreactivity risk'
        elif comprehensive_risk < 0.8:
            risk_level = 'High'
            recommendation = 'High polyreactivity risk, significant optimization needed'
        else:
            risk_level = 'Very High'
            recommendation = 'Very high polyreactivity risk, likely to fail developability screens'
        
        return {
            'comprehensive_risk_score': comprehensive_risk,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'charge_risk': charge_risk,
            'hydrophobicity_risk': hydro_risk,
            'paratope_risk': paratope_risk
        }


def map_psr_psp_assay_results(sequence_features: Dict[str, Union[float, Dict]]) -> Dict[str, Union[float, str]]:
    """
    Map sequence features to PSR/PSP assay results using decision rules.
    
    Args:
        sequence_features (Dict): Dictionary of sequence features
        
    Returns:
        Dict: Mapped assay results and risk assessment
    """
    # Initialize evaluator
    evaluator = PSRPSPEvaluator()
    
    # Extract relevant features
    charge_features = {
        'vh_net_charge': sequence_features.get('vh_net_charge', 0.0),
        'vl_net_charge': sequence_features.get('vl_net_charge', 0.0),
        'charge_imbalance': sequence_features.get('charge_imbalance', 0.0)
    }
    
    hydrophobicity_features = {
        'patch_density': sequence_features.get('hydrophobic_patch_density', 0.0),
        'surface_exposed_hydrophobicity': sequence_features.get('surface_exposed_hydrophobicity', 0.0)
    }
    
    paratope_features = {
        'paratope_entropy': sequence_features.get('combined_paratope_entropy', 0.0),
        'aromatic_fraction': sequence_features.get('paratope_aromatic_fraction', 0.0)
    }
    
    # Evaluate comprehensive risk
    risk_assessment = evaluator.evaluate_comprehensive_risk(
        charge_features, hydrophobicity_features, paratope_features
    )
    
    # Map to assay-like results
    # This is a simplified mapping - in practice, this would be based on
    # empirical data and more sophisticated models
    risk_score = risk_assessment['comprehensive_risk_score']
    
    # Simulate PSR/PSP assay results
    # Higher risk scores correspond to higher polyreactivity/self-association
    if risk_score < 0.3:
        psr_result = 'Negative'  # Low polyreactivity
        psp_result = 'Negative'  # Low self-association
        binding_score = np.random.normal(1.5, 0.5)  # Low non-specific binding
    elif risk_score < 0.6:
        psr_result = 'Weak Positive'  # Moderate polyreactivity
        psp_result = 'Weak Positive'  # Moderate self-association
        binding_score = np.random.normal(3.0, 1.0)  # Moderate non-specific binding
    elif risk_score < 0.8:
        psr_result = 'Positive'  # High polyreactivity
        psp_result = 'Positive'  # High self-association
        binding_score = np.random.normal(5.0, 1.5)  # High non-specific binding
    else:
        psr_result = 'Strong Positive'  # Very high polyreactivity
        psp_result = 'Strong Positive'  # Very high self-association
        binding_score = np.random.normal(7.5, 2.0)  # Very high non-specific binding
    
    # Ensure binding score is positive
    binding_score = max(0.0, binding_score)
    
    return {
        'psr_result': psr_result,
        'psp_result': psp_result,
        'binding_score': binding_score,
        'risk_score': risk_score,
        'risk_level': risk_assessment['risk_level'],
        'recommendation': risk_assessment['recommendation'],
        'detailed_risk_assessment': risk_assessment
    }


def create_psr_psp_decision_tree() -> Dict:
    """
    Create a decision tree for PSR/PSP assay mapping.
    
    Returns:
        Dict: Decision tree structure
    """
    return {
        'root': {
            'question': 'What is the comprehensive risk score?',
            'conditions': [
                {
                    'condition': 'risk_score < 0.3',
                    'result': {
                        'psr_result': 'Negative',
                        'psp_result': 'Negative',
                        'binding_score_range': '0-2',
                        'recommendation': 'Antibody likely to have good developability'
                    }
                },
                {
                    'condition': '0.3 <= risk_score < 0.6',
                    'result': {
                        'psr_result': 'Weak Positive',
                        'psp_result': 'Weak Positive',
                        'binding_score_range': '2-4',
                        'recommendation': 'Consider optimization to reduce polyreactivity risk'
                    }
                },
                {
                    'condition': '0.6 <= risk_score < 0.8',
                    'result': {
                        'psr_result': 'Positive',
                        'psp_result': 'Positive',
                        'binding_score_range': '4-6',
                        'recommendation': 'High polyreactivity risk, significant optimization needed'
                    }
                },
                {
                    'condition': 'risk_score >= 0.8',
                    'result': {
                        'psr_result': 'Strong Positive',
                        'psp_result': 'Strong Positive',
                        'binding_score_range': '6-10',
                        'recommendation': 'Very high polyreactivity risk, likely to fail developability screens'
                    }
                }
            ]
        }
    }


def main():
    """
    Example usage of the PSR/PSP assay mapping implementation.
    ""
    # Example sequence features (simplified)
    sequence_features = {
        'vh_net_charge': 1.5,
        'vl_net_charge': -0.5,
        'charge_imbalance': 2.0,
        'hydrophobic_patch_density': 0.08,
        'surface_exposed_hydrophobicity': 0.25,
        'combined_paratope_entropy': 3.2,
        'paratope_aromatic_fraction': 0.15
    }
    
    # Map to PSR/PSP assay results
    assay_results = map_psr_psp_assay_results(sequence_features)
    
    print("PSR/PSP Assay Mapping Results:")
    print(f"PSR Result: {assay_results['psr_result']}")
    print(f"PSP Result: {assay_results['psp_result']}")
    print(f"Binding Score: {assay_results['binding_score']:.2f}")
    print(f"Risk Score: {assay_results['risk_score']:.3f}")
    print(f"Risk Level: {assay_results['risk_level']}")
    print(f"Recommendation: {assay_results['recommendation']}")


if __name__ == "__main__":
    main()
