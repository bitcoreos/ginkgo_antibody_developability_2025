"""
PSR/PSP Assay Mapping Module

This module implements mapping of polyreactivity features to PSR/PSP assay data and creation of decision rules.
"""

import numpy as np
from typing import Dict, List, Tuple, Union
from collections import Counter

# Assay thresholds for decision rules
PSR_THRESHOLD = 0.5  # Threshold for Polyreactivity Specificity Ratio
PSP_THRESHOLD = 0.3  # Threshold for Polyreactivity Specificity Potential


class AssayMapper:
    """
    Mapper for PSR/PSP assay data and decision rules.
    """
    
    def __init__(self):
        """
        Initialize the assay mapper.
        """
        pass
    
    def map_polyreactivity_features(self, sequence: str, 
                               charge_imbalance_score: float,
                               clustering_risk_score: float,
                               binding_potential: float,
                               dynamics_risk_score: float) -> Dict[str, Union[int, float, Dict]]:
        """
        Map polyreactivity features to assay data and create decision rules.
        
        Args:
            sequence (str): Amino acid sequence
            charge_imbalance_score (float): Charge imbalance score from ChargeImbalanceAnalyzer
            clustering_risk_score (float): Clustering risk score from ResidueClusteringAnalyzer
            binding_potential (float): Binding potential from HydrophobicPatchAnalyzer
            dynamics_risk_score (float): Dynamics risk score from ParatopeDynamicsAnalyzer
            
        Returns:
            Dict: Assay mapping results
        """
        sequence = sequence.upper()
        length = len(sequence)
        
        if length == 0:
            return {
                'sequence': sequence,
                'length': 0,
                'assay_features': {},
                'psr_score': 0.0,
                'psp_score': 0.0,
                'decision_rules': {},
                'mapping_complete': True
            }
        
        # Create feature vector for assay mapping
        assay_features = {
            'charge_imbalance': charge_imbalance_score,
            'clustering_risk': clustering_risk_score,
            'binding_potential': binding_potential,
            'dynamics_risk': dynamics_risk_score
        }
        
        # Calculate PSR (Polyreactivity Specificity Ratio)
        # Lower scores indicate better specificity (less polyreactivity)
        psr_score = self._calculate_psr(assay_features)
        
        # Calculate PSP (Polyreactivity Specificity Potential)
        # Lower scores indicate better potential for specificity
        psp_score = self._calculate_psp(assay_features)
        
        # Apply decision rules
        decision_rules = self._apply_decision_rules(psr_score, psp_score, assay_features)
        
        return {
            'sequence': sequence,
            'length': length,
            'assay_features': assay_features,
            'psr_score': psr_score,
            'psp_score': psp_score,
            'decision_rules': decision_rules,
            'mapping_complete': True
        }
    
    def _calculate_psr(self, assay_features: Dict[str, float]) -> float:
        """
        Calculate Polyreactivity Specificity Ratio (PSR).
        
        Args:
            assay_features (Dict[str, float]): Polyreactivity features
            
        Returns:
            float: PSR score (0-1, lower is better)
        """
        # PSR is a weighted combination of polyreactivity features
        # Higher values indicate higher polyreactivity (lower specificity)
        psr = (
            0.3 * assay_features['charge_imbalance'] +
            0.3 * assay_features['clustering_risk'] +
            0.2 * assay_features['binding_potential'] +
            0.2 * assay_features['dynamics_risk']
        )
        
        return min(1.0, psr)  # Clamp to 0-1 range
    
    def _calculate_psp(self, assay_features: Dict[str, float]) -> float:
        """
        Calculate Polyreactivity Specificity Potential (PSP).
        
        Args:
            assay_features (Dict[str, float]): Polyreactivity features
            
        Returns:
            float: PSP score (0-1, lower is better)
        """
        # PSP is a modified version of PSR that considers potential for improvement
        # It's calculated as PSR adjusted by a factor that represents how difficult it is to improve
        
        # Calculate base PSR
        psr = self._calculate_psr(assay_features)
        
        # Calculate "improvability factor" (0-1, higher means easier to improve)
        # Based on the idea that extreme values are harder to improve
        improvability = 1.0 - psr  # Inverse relationship
        
        # PSP is PSR adjusted by improvability
        psp = psr * (1.0 + 0.5 * (1.0 - improvability))
        
        return min(1.0, psp)  # Clamp to 0-1 range
    
    def _apply_decision_rules(self, psr_score: float, psp_score: float, 
                         assay_features: Dict[str, float]) -> Dict[str, Union[str, bool]]:
        """
        Apply decision rules based on PSR/PSP scores and features.
        
        Args:
            psr_score (float): PSR score
            psp_score (float): PSP score
            assay_features (Dict[str, float]): Polyreactivity features
            
        Returns:
            Dict: Decision rules results
        """
        # Decision rule 1: Overall polyreactivity assessment
        if psr_score < 0.2:
            polyreactivity_assessment = "Low polyreactivity - favorable for specificity"
        elif psr_score < 0.4:
            polyreactivity_assessment = "Moderate polyreactivity - generally acceptable"
        elif psr_score < 0.6:
            polyreactivity_assessment = "High polyreactivity - may affect specificity"
        else:
            polyreactivity_assessment = "Very high polyreactivity - likely to cause specificity issues"
        
        # Decision rule 2: Developability risk assessment
        if psp_score < 0.3:
            developability_risk = "Low developability risk"
        elif psp_score < 0.5:
            developability_risk = "Moderate developability risk"
        elif psp_score < 0.7:
            developability_risk = "High developability risk"
        else:
            developability_risk = "Very high developability risk"
        
        # Decision rule 3: Recommended actions
        recommended_actions = []
        
        if assay_features['charge_imbalance'] > 0.3:
            recommended_actions.append("Consider charge balancing modifications")
        
        if assay_features['clustering_risk'] > 0.3:
            recommended_actions.append("Consider breaking up residue clusters")
        
        if assay_features['binding_potential'] > 0.4:
            recommended_actions.append("Consider reducing hydrophobic patches")
        
        if assay_features['dynamics_risk'] > 0.4:
            recommended_actions.append("Consider stabilizing paratope dynamics")
        
        if not recommended_actions:
            recommended_actions.append("No immediate modifications recommended")
        
        # Decision rule 4: Priority ranking
        if psr_score > 0.6 or psp_score > 0.7:
            priority = "High - immediate attention required"
        elif psr_score > 0.4 or psp_score > 0.5:
            priority = "Medium - should be addressed"
        else:
            priority = "Low - monitor but no immediate action needed"
        
        return {
            'polyreactivity_assessment': polyreactivity_assessment,
            'developability_risk': developability_risk,
            'recommended_actions': recommended_actions,
            'priority': priority,
            'rules_applied': True
        }
    
    def generate_assay_report(self, sequence: str,
                         charge_imbalance_score: float,
                         clustering_risk_score: float,
                         binding_potential: float,
                         dynamics_risk_score: float) -> Dict[str, Union[str, float, Dict]]:
        """
        Generate a comprehensive assay report.
        
        Args:
            sequence (str): Amino acid sequence
            charge_imbalance_score (float): Charge imbalance score
            clustering_risk_score (float): Clustering risk score
            binding_potential (float): Binding potential
            dynamics_risk_score (float): Dynamics risk score
            
        Returns:
            Dict: Comprehensive assay report
        """
        # Map polyreactivity features
        assay_mapping = self.map_polyreactivity_features(
            sequence, charge_imbalance_score, clustering_risk_score,
            binding_potential, dynamics_risk_score
        )
        
        # Extract key metrics
        psr_score = assay_mapping['psr_score']
        psp_score = assay_mapping['psp_score']
        decision_rules = assay_mapping['decision_rules']
        
        # Generate summary
        actions_text = "\n".join([f"- {action}" for action in decision_rules['recommended_actions']])
        summary = f"""
Polyreactivity Assay Report
=========================

PSR Score: {psr_score:.3f} ({decision_rules['polyreactivity_assessment']})
PSP Score: {psp_score:.3f} ({decision_rules['developability_risk']})
Priority: {decision_rules['priority']}

Recommended Actions:
{actions_text}
"""
        
        return {
            'sequence': sequence,
            'psr_score': psr_score,
            'psp_score': psp_score,
            'assay_features': assay_mapping['assay_features'],
            'decision_rules': decision_rules,
            'summary': summary,
            'report_complete': True
        }


def main():
    """
    Example usage of the assay mapper.
    """
    # Example sequence
    sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    
    # Example feature scores (from previous analyses)
    charge_imbalance_score = 0.013
    clustering_risk_score = 0.760
    binding_potential = 0.311
    dynamics_risk_score = 0.361
    
    # Create mapper
    mapper = AssayMapper()
    
    # Map polyreactivity features
    assay_mapping = mapper.map_polyreactivity_features(
        sequence, charge_imbalance_score, clustering_risk_score,
        binding_potential, dynamics_risk_score
    )
    
    print("PSR/PSP Assay Mapping:")
    print(f"  PSR Score: {assay_mapping['psr_score']:.3f}")
    print(f"  PSP Score: {assay_mapping['psp_score']:.3f}")
    
    # Print decision rules
    rules = assay_mapping['decision_rules']
    print("\nDecision Rules:")
    print(f"  Polyreactivity Assessment: {rules['polyreactivity_assessment']}")
    print(f"  Developability Risk: {rules['developability_risk']}")
    print(f"  Priority: {rules['priority']}")
    print("  Recommended Actions:")
    for action in rules['recommended_actions']:
        print(f"    - {action}")
    
    # Generate comprehensive assay report
    assay_report = mapper.generate_assay_report(
        sequence, charge_imbalance_score, clustering_risk_score,
        binding_potential, dynamics_risk_score
    )
    print("\nAssay Report Summary:")
    print(assay_report['summary'])


if __name__ == "__main__":
    main()
