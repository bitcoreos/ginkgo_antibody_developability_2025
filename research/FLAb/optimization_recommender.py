"""
Optimization Recommender Implementation

This module implements fragment optimization recommendations for the FLAb framework.
"""

import numpy as np
from typing import Dict, List, Tuple

# Amino acid properties for optimization
HYDROPHOBIC_AA = 'AILMFWV'
CHARGED_AA = 'DEKR'
POLAR_AA = 'NQST'


class OptimizationRecommender:
    """
    Fragment optimization recommendations.
    """
    
    def __init__(self):
        """
        Initialize the optimization recommender.
        """
        pass
    
    def recommend_fragment_design(self, fragment_sequence: str, analysis_results: Dict) -> Dict:
        """
        Recommend fragment design improvements based on analysis results.
        
        Args:
            fragment_sequence (str): Fragment sequence to analyze
            analysis_results (Dict): Previous analysis results
            
        Returns:
            Dict: Design recommendations
        """
        # Validate inputs
        if not fragment_sequence or not analysis_results:
            return {
                'sequence': fragment_sequence,
                'recommendations': [],
                'recommendation_complete': False,
                'error': 'Invalid inputs'
            }
        
        # Generate recommendations
        recommendations = []
        
        # Check sequence analysis for recommendations
        if 'sequence_analysis' in analysis_results:
            seq_analysis = analysis_results['sequence_analysis']
            recommendations.extend(self._recommend_sequence_optimizations(seq_analysis))
        
        # Check structural analysis for recommendations
        if 'structural_analysis' in analysis_results:
            struct_analysis = analysis_results['structural_analysis']
            recommendations.extend(self._recommend_structural_optimizations(struct_analysis))
        
        # Check physicochemical analysis for recommendations
        if 'physicochemical_analysis' in analysis_results:
            phys_analysis = analysis_results['physicochemical_analysis']
            recommendations.extend(self._recommend_physicochemical_optimizations(phys_analysis))
        
        # Check stability analysis for recommendations
        if 'stability_analysis' in analysis_results:
            stab_analysis = analysis_results['stability_analysis']
            recommendations.extend(self._recommend_stability_optimizations(stab_analysis))
        
        return {
            'sequence': fragment_sequence,
            'recommendations': recommendations,
            'recommendation_count': len(recommendations),
            'recommendation_complete': True
        }
    
    def _recommend_sequence_optimizations(self, seq_analysis: Dict) -> List[Dict]:
        """
        Recommend sequence optimizations based on sequence analysis.
        
        Args:
            seq_analysis (Dict): Sequence analysis results
            
        Returns:
            List[Dict]: Sequence optimization recommendations
        """
        recommendations = []
        
        # Check net charge
        net_charge = seq_analysis.get('net_charge', 0)
        if abs(net_charge) > 5:
            recommendations.append({
                'type': 'sequence',
                'category': 'charge_optimization',
                'priority': 'high' if abs(net_charge) > 10 else 'medium',
                'description': f'Net charge is {net_charge}. Consider reducing charge imbalance.',
                'suggestion': 'Replace charged residues with neutral ones in non-critical regions.'
            })
        
        # Check hydrophobicity
        hydrophobicity = seq_analysis.get('hydrophobicity', 0)
        if hydrophobicity > 0.5:
            recommendations.append({
                'type': 'sequence',
                'category': 'hydrophobicity_optimization',
                'priority': 'high' if hydrophobicity > 0.7 else 'medium',
                'description': f'Hydrophobicity is high ({hydrophobicity:.2f}). May affect solubility.',
                'suggestion': 'Replace some hydrophobic residues with polar/charged ones in surface-exposed regions.'
            })
        elif hydrophobicity < 0.2:
            recommendations.append({
                'type': 'sequence',
                'category': 'hydrophobicity_optimization',
                'priority': 'medium',
                'description': f'Hydrophobicity is low ({hydrophobicity:.2f}). May affect stability.',
                'suggestion': 'Add some hydrophobic residues to improve core stability.'
            })
        
        # Check polarity
        polarity = seq_analysis.get('polarity', 0)
        if polarity > 0.6:
            recommendations.append({
                'type': 'sequence',
                'category': 'polarity_optimization',
                'priority': 'medium',
                'description': f'Polarity is high ({polarity:.2f}). May affect stability.',
                'suggestion': 'Balance polar residues with hydrophobic ones for better stability.'
            })
        
        return recommendations
    
    def _recommend_structural_optimizations(self, struct_analysis: Dict) -> List[Dict]:
        """
        Recommend structural optimizations based on structural analysis.
        
        Args:
            struct_analysis (Dict): Structural analysis results
            
        Returns:
            List[Dict]: Structural optimization recommendations
        """
        recommendations = []
        
        # Check helix propensity
        helix_propensity = struct_analysis.get('helix_propensity', 0)
        if helix_propensity > 0.6:
            recommendations.append({
                'type': 'structural',
                'category': 'secondary_structure_optimization',
                'priority': 'medium',
                'description': f'High helix propensity ({helix_propensity:.2f}). May affect flexibility.',
                'suggestion': 'Introduce proline or glycine residues to break up long helices if flexibility is needed.'
            })
        
        # Check sheet propensity
        sheet_propensity = struct_analysis.get('sheet_propensity', 0)
        if sheet_propensity > 0.6:
            recommendations.append({
                'type': 'structural',
                'category': 'secondary_structure_optimization',
                'priority': 'medium',
                'description': f'High sheet propensity ({sheet_propensity:.2f}). May affect solubility.',
                'suggestion': 'Modify regions with high sheet propensity to improve solubility if needed.'
            })
        
        # Check flexibility
        flexibility = struct_analysis.get('flexibility', 0)
        if flexibility < 0.1:
            recommendations.append({
                'type': 'structural',
                'category': 'flexibility_optimization',
                'priority': 'medium',
                'description': f'Low flexibility ({flexibility:.2f}). May affect binding.',
                'suggestion': 'Introduce glycine or proline residues to increase flexibility in loop regions.'
            })
        
        return recommendations
    
    def _recommend_physicochemical_optimizations(self, phys_analysis: Dict) -> List[Dict]:
        """
        Recommend physicochemical optimizations based on physicochemical analysis.
        
        Args:
            phys_analysis (Dict): Physicochemical analysis results
            
        Returns:
            List[Dict]: Physicochemical optimization recommendations
        """
        recommendations = []
        
        # Check isoelectric point
        pI = phys_analysis.get('isoelectric_point', 7.0)
        if pI < 5.0:
            recommendations.append({
                'type': 'physicochemical',
                'category': 'pI_optimization',
                'priority': 'high' if pI < 4.0 else 'medium',
                'description': f'Low pI ({pI:.2f}). May cause problems in physiological conditions.',
                'suggestion': 'Add basic residues or remove acidic residues to increase pI.'
            })
        elif pI > 9.0:
            recommendations.append({
                'type': 'physicochemical',
                'category': 'pI_optimization',
                'priority': 'high' if pI > 10.0 else 'medium',
                'description': f'High pI ({pI:.2f}). May cause problems in physiological conditions.',
                'suggestion': 'Add acidic residues or remove basic residues to decrease pI.'
            })
        
        # Check GRAVY (hydrophobicity)
        gravy = phys_analysis.get('gravy', 0)
        if gravy > 1.0:
            recommendations.append({
                'type': 'physicochemical',
                'category': 'hydrophobicity_optimization',
                'priority': 'high' if gravy > 2.0 else 'medium',
                'description': f'High hydrophobicity (GRAVY={gravy:.2f}). May affect solubility.',
                'suggestion': 'Replace some hydrophobic residues with hydrophilic ones to improve solubility.'
            })
        elif gravy < -1.0:
            recommendations.append({
                'type': 'physicochemical',
                'category': 'hydrophobicity_optimization',
                'priority': 'medium',
                'description': f'Low hydrophobicity (GRAVY={gravy:.2f}). May affect membrane stability.',
                'suggestion': 'Add some hydrophobic residues to improve stability in membrane environments.'
            })
        
        return recommendations
    
    def _recommend_stability_optimizations(self, stab_analysis: Dict) -> List[Dict]:
        """
        Recommend stability optimizations based on stability analysis.
        
        Args:
            stab_analysis (Dict): Stability analysis results
            
        Returns:
            List[Dict]: Stability optimization recommendations
        """
        recommendations = []
        
        # Check instability score
        instability_score = stab_analysis.get('instability_score', 0)
        if instability_score > 0.3:
            recommendations.append({
                'type': 'stability',
                'category': 'instability_optimization',
                'priority': 'high' if instability_score > 0.5 else 'medium',
                'description': f'High instability score ({instability_score:.2f}). May degrade quickly.',
                'suggestion': 'Reduce charged residues and proline content to improve stability.'
            })
        
        # Check aggregation propensity
        aggregation_propensity = stab_analysis.get('aggregation_propensity', 0)
        if aggregation_propensity > 0.2:
            recommendations.append({
                'type': 'stability',
                'category': 'aggregation_optimization',
                'priority': 'high' if aggregation_propensity > 0.4 else 'medium',
                'description': f'High aggregation propensity ({aggregation_propensity:.2f}). May form aggregates.',
                'suggestion': 'Reduce hydrophobic patches and aromatic residues to decrease aggregation.'
            })
        
        # Check thermal stability
        thermal_stability = stab_analysis.get('thermal_stability', 0.5)
        if thermal_stability < 0.3:
            recommendations.append({
                'type': 'stability',
                'category': 'thermal_stability_optimization',
                'priority': 'high' if thermal_stability < 0.1 else 'medium',
                'description': f'Low thermal stability ({thermal_stability:.2f}). May denature easily.',
                'suggestion': 'Increase proline content and optimize hydrophobic core to improve thermal stability.'
            })
        
        return recommendations
    
    def suggest_sequence_modifications(self, fragment_sequence: str, developability_results: Dict) -> Dict:
        """
        Suggest specific sequence modifications for improved developability.
        
        Args:
            fragment_sequence (str): Fragment sequence to analyze
            developability_results (Dict): Previous developability results
            
        Returns:
            Dict: Sequence modification suggestions
        """
        # Validate inputs
        if not fragment_sequence or not developability_results:
            return {
                'sequence': fragment_sequence,
                'modifications': [],
                'suggestion_complete': False,
                'error': 'Invalid inputs'
            }
        
        # Generate modifications
        modifications = []
        
        # Suggest modifications based on developability predictions
        if 'solubility_score' in developability_results:
            solubility = developability_results['solubility_score']
            if solubility < 0.3:
                # Suggest solubility improvements
                modifications.extend(self._suggest_solubility_modifications(fragment_sequence))
        
        if 'aggregation_propensity' in developability_results:
            aggregation = developability_results['aggregation_propensity']
            if aggregation > 0.7:
                # Suggest aggregation reduction
                modifications.extend(self._suggest_aggregation_modifications(fragment_sequence))
        
        if 'expression_level' in developability_results:
            expression = developability_results['expression_level']
            if expression < 0.3:
                # Suggest expression improvements
                modifications.extend(self._suggest_expression_modifications(fragment_sequence))
        
        return {
            'sequence': fragment_sequence,
            'modifications': modifications,
            'modification_count': len(modifications),
            'suggestion_complete': True
        }
    
    def _suggest_solubility_modifications(self, sequence: str) -> List[Dict]:
        """
        Suggest modifications to improve solubility.
        
        Args:
            sequence (str): Fragment sequence
            
        Returns:
            List[Dict]: Solubility modification suggestions
        """
        modifications = []
        sequence = sequence.upper()
        
        # Look for hydrophobic patches to modify
        for i in range(len(sequence) - 4):
            window = sequence[i:i+5]
            hydrophobic_count = sum(1 for aa in window if aa in HYDROPHOBIC_AA)
            
            if hydrophobic_count >= 3:
                # Suggest replacing one hydrophobic residue with a polar one
                for j, aa in enumerate(window):
                    if aa in HYDROPHOBIC_AA:
                        modifications.append({
                            'position': i + j,
                            'original_aa': aa,
                            'suggested_aa': 'S',  # Serine as a polar replacement
                            'type': 'solubility_improvement',
                            'reason': 'Reduce hydrophobic patch'
                        })
                        break  # Only suggest one change per window
        
        return modifications
    
    def _suggest_aggregation_modifications(self, sequence: str) -> List[Dict]:
        """
        Suggest modifications to reduce aggregation.
        
        Args:
            sequence (str): Fragment sequence
            
        Returns:
            List[Dict]: Aggregation reduction suggestions
        """
        modifications = []
        sequence = sequence.upper()
        
        # Look for aromatic residues to modify
        for i, aa in enumerate(sequence):
            if aa in 'FWY':  # Aromatic residues
                modifications.append({
                    'position': i,
                    'original_aa': aa,
                    'suggested_aa': 'A',  # Alanine as a less aromatic replacement
                    'type': 'aggregation_reduction',
                    'reason': 'Reduce aromatic interactions'
                })
        
        return modifications
    
    def _suggest_expression_modifications(self, sequence: str) -> List[Dict]:
        """
        Suggest modifications to improve expression.
        
        Args:
            sequence (str): Fragment sequence
            
        Returns:
            List[Dict]: Expression improvement suggestions
        """
        modifications = []
        sequence = sequence.upper()
        
        # Look for rare codons (represented by rare amino acids)
        # In this simplified model, we'll consider certain amino acids as "rare"
        rare_aa = 'CMW'  # Cysteine, Methionine, Tryptophan are often rare
        
        for i, aa in enumerate(sequence):
            if aa in rare_aa:
                modifications.append({
                    'position': i,
                    'original_aa': aa,
                    'suggested_aa': 'S' if aa == 'C' else 'L',  # Serine for Cysteine, Leucine for others
                    'type': 'expression_improvement',
                    'reason': 'Replace rare amino acid'
                })
        
        return modifications
    
    def recommend_structural_optimization(self, fragment_sequence: str, structural_results: Dict) -> Dict:
        """
        Recommend structural optimization strategies.
        
        Args:
            fragment_sequence (str): Fragment sequence to analyze
            structural_results (Dict): Previous structural analysis results
            
        Returns:
            Dict: Structural optimization recommendations
        """
        # Validate inputs
        if not fragment_sequence or not structural_results:
            return {
                'sequence': fragment_sequence,
                'optimization_strategies': [],
                'recommendation_complete': False,
                'error': 'Invalid inputs'
            }
        
        # Generate optimization strategies
        strategies = []
        
        # Analyze secondary structure propensities
        helix_propensity = structural_results.get('helix_propensity', 0)
        sheet_propensity = structural_results.get('sheet_propensity', 0)
        turn_propensity = structural_results.get('turn_propensity', 0)
        
        # Recommend strategies based on structural analysis
        if helix_propensity > 0.5:
            strategies.append({
                'strategy': 'helix_breaking',
                'description': 'High helix propensity may limit flexibility',
                'approach': 'Introduce proline or glycine residues in helical regions',
                'priority': 'medium'
            })
        
        if sheet_propensity > 0.5:
            strategies.append({
                'strategy': 'sheet_reduction',
                'description': 'High sheet propensity may affect solubility',
                'approach': 'Modify sheet-prone regions with proline or charged residues',
                'priority': 'medium'
            })
        
        if turn_propensity < 0.2:
            strategies.append({
                'strategy': 'turn_promotion',
                'description': 'Low turn propensity may limit structural diversity',
                'approach': 'Introduce glycine or proline to promote turns',
                'priority': 'low'
            })
        
        # Flexibility recommendations
        flexibility = structural_results.get('flexibility', 0)
        if flexibility < 0.1:
            strategies.append({
                'strategy': 'flexibility_increase',
                'description': 'Low flexibility may limit binding ability',
                'approach': 'Introduce glycine in loop regions to increase flexibility',
                'priority': 'medium'
            })
        
        return {
            'sequence': fragment_sequence,
            'optimization_strategies': strategies,
            'strategy_count': len(strategies),
            'recommendation_complete': True
        }


def main():
    """
    Example usage of the optimization recommender.
    """
    # Example fragment sequence
    fragment_sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVARIYYSGSTNYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    
    # Example analysis results (simplified)
    analysis_results = {
        'sequence_analysis': {
            'net_charge': 2,
            'hydrophobicity': 0.45,
            'polarity': 0.35
        },
        'structural_analysis': {
            'helix_propensity': 0.4,
            'sheet_propensity': 0.3,
            'turn_propensity': 0.3,
            'flexibility': 0.2
        },
        'physicochemical_analysis': {
            'isoelectric_point': 8.2,
            'gravy': 0.5
        },
        'stability_analysis': {
            'instability_score': 0.25,
            'aggregation_propensity': 0.3,
            'thermal_stability': 0.6
        }
    }
    
    # Example developability results (simplified)
    developability_results = {
        'solubility_score': 0.4,
        'expression_level': 0.6,
        'aggregation_propensity': 0.35
    }
    
    # Example structural results (simplified)
    structural_results = {
        'helix_propensity': 0.4,
        'sheet_propensity': 0.3,
        'turn_propensity': 0.3,
        'flexibility': 0.2
    }
    
    # Create recommender
    recommender = OptimizationRecommender()
    
    # Recommend fragment design
    design_recommendations = recommender.recommend_fragment_design(fragment_sequence, analysis_results)
    print("Design Recommendations:")
    print(f"  Recommendation Count: {design_recommendations['recommendation_count']}")
    for i, rec in enumerate(design_recommendations['recommendations']):
        print(f"  {i+1}. {rec['type']} - {rec['category']} ({rec['priority']}): {rec['description']}")
        print(f"     Suggestion: {rec['suggestion']}")
    
    # Suggest sequence modifications
    sequence_modifications = recommender.suggest_sequence_modifications(fragment_sequence, developability_results)
    print("\nSequence Modifications:")
    print(f"  Modification Count: {sequence_modifications['modification_count']}")
    for i, mod in enumerate(sequence_modifications['modifications']):
        print(f"  {i+1}. Position {mod['position']}: {mod['original_aa']} -> {mod['suggested_aa']}")
        print(f"     Type: {mod['type']}, Reason: {mod['reason']}")
    
    # Recommend structural optimization
    structural_optimization = recommender.recommend_structural_optimization(fragment_sequence, structural_results)
    print("\nStructural Optimization Strategies:")
    print(f"  Strategy Count: {structural_optimization['strategy_count']}")
    for i, strategy in enumerate(structural_optimization['optimization_strategies']):
        print(f"  {i+1}. {strategy['strategy']} ({strategy['priority']}): {strategy['description']}")
        print(f"     Approach: {strategy['approach']}")


if __name__ == "__main__":
    main()
