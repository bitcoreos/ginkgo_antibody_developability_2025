"""
Optimization Recommender Module for FLAb Framework

This module provides design improvements and sequence modifications for antibody fragments.
"""


class OptimizationRecommender:
    """
    Class for providing optimization recommendations for antibody fragments.
    """
    
    def __init__(self):
        """
        Initialize the OptimizationRecommender.
        """
        pass
    
    def generate_recommendations(self, fragment_analysis, developability_predictions):
        """
        Generate optimization recommendations based on fragment analysis and developability predictions.
        
        Args:
            fragment_analysis (dict): Fragment analysis results from FragmentAnalyzer
            developability_predictions (dict): Developability predictions from DevelopabilityPredictor
            
        Returns:
            dict: Optimization recommendations
        """
        recommendations = {
            'sequence_modifications': [],
            'structural_strategies': [],
            'design_improvements': [],
            'priority': 'medium'
        }
        
        # Analyze developability predictions
        if developability_predictions['overall_score'] < 0.3:
            recommendations['priority'] = 'high'
        elif developability_predictions['overall_score'] > 0.7:
            recommendations['priority'] = 'low'
        
        # Generate sequence modification recommendations
        sequence_modifications = self._recommend_sequence_modifications(fragment_analysis)
        recommendations['sequence_modifications'] = sequence_modifications
        
        # Generate structural optimization strategies
        structural_strategies = self._recommend_structural_strategies(fragment_analysis)
        recommendations['structural_strategies'] = structural_strategies
        
        # Generate design improvement recommendations
        design_improvements = self._recommend_design_improvements(developability_predictions)
        recommendations['design_improvements'] = design_improvements
        
        return recommendations
    
    def _recommend_sequence_modifications(self, fragment_analysis):
        """
        Recommend sequence modifications based on fragment analysis.
        
        Args:
            fragment_analysis (dict): Fragment analysis results
            
        Returns:
            list: Sequence modification recommendations
        """
        recommendations = []
        
        # Extract relevant data
        composition = fragment_analysis.get('composition', {})
        phys_props = fragment_analysis.get('physicochemical_properties', {})
        stability = fragment_analysis.get('stability', {})
        
        if composition:
            total_aa = composition.get('total_amino_acids', 1)
            if total_aa > 0:
                # Check hydrophobicity
                hydrophobic_ratio = composition['groups'].get('hydrophobic', 0) / total_aa
                if hydrophobic_ratio > 0.5:
                    recommendations.append({
                        'type': 'reduce_hydrophobicity',
                        'description': f'High hydrophobic content ({hydrophobic_ratio:.2%}). Consider replacing some hydrophobic residues with polar ones.',
                        'priority': 'medium'
                    })
                
                # Check aromatic content
                aromatic_ratio = composition['groups'].get('aromatic', 0) / total_aa
                if aromatic_ratio > 0.15:
                    recommendations.append({
                        'type': 'reduce_aromatic_content',
                        'description': f'High aromatic content ({aromatic_ratio:.2%}). Consider reducing aromatic residues to decrease aggregation propensity.',
                        'priority': 'medium'
                    })
                
                # Check charged content
                charged_ratio = composition['groups'].get('charged', 0) / total_aa
                if charged_ratio < 0.15:
                    recommendations.append({
                        'type': 'increase_charged_content',
                        'description': f'Low charged content ({charged_ratio:.2%}). Consider adding charged residues to improve solubility.',
                        'priority': 'medium'
                    })
        
        if phys_props:
            # Check hydrophobicity
            hydrophobicity = phys_props.get('hydrophobicity', 0)
            if hydrophobicity < -0.5:
                recommendations.append({
                    'type': 'adjust_hydrophobicity',
                    'description': f'Low hydrophobicity ({hydrophobicity:.2f}). Consider adjusting to improve membrane interaction if needed.',
                    'priority': 'low'
                })
            
            # Check isoelectric point
            pi = phys_props.get('isoelectric_point', 7.0)
            if pi < 6.0 or pi > 8.5:
                recommendations.append({
                    'type': 'adjust_pi',
                    'description': f'Extreme pI ({pi:.2f}). Consider adjusting to improve solubility at physiological pH.',
                    'priority': 'high'
                })
        
        if stability:
            # Check aggregation propensity
            aggregation_propensity = stability.get('aggregation_propensity', 0)
            if aggregation_propensity > 0.7:
                recommendations.append({
                    'type': 'reduce_aggregation_propensity',
                    'description': f'High aggregation propensity ({aggregation_propensity:.2f}). Consider sequence modifications to reduce hydrophobic patches.',
                    'priority': 'high'
                })
            
            # Check thermal stability
            thermal_stability = stability.get('thermal_stability', 0)
            if thermal_stability < 0.3:
                recommendations.append({
                    'type': 'improve_thermal_stability',
                    'description': f'Low thermal stability ({thermal_stability:.2f}). Consider adding disulfide bonds or stabilizing motifs.',
                    'priority': 'medium'
                })
        
        return recommendations
    
    def _recommend_structural_strategies(self, fragment_analysis):
        """
        Recommend structural optimization strategies based on fragment analysis.
        
        Args:
            fragment_analysis (dict): Fragment analysis results
            
        Returns:
            list: Structural optimization strategies
        """
        strategies = []
        
        # Extract relevant data
        composition = fragment_analysis.get('composition', {})
        stability = fragment_analysis.get('stability', {})
        
        if composition:
            total_aa = composition.get('total_amino_acids', 1)
            if total_aa > 0:
                # Check for disulfide bond potential
                cysteine_count = composition.get('counts', {}).get('C', 0)
                if cysteine_count >= 2:
                    strategies.append({
                        'type': 'disulfide_bond_formation',
                        'description': f'Found {cysteine_count} cysteine residues. Consider engineering disulfide bonds to improve stability.',
                        'priority': 'medium'
                    })
                elif cysteine_count == 0:
                    strategies.append({
                        'type': 'add_disulfide_bonds',
                        'description': 'No cysteine residues found. Consider adding cysteines for potential disulfide bond formation.',
                        'priority': 'low'
                    })
        
        if stability:
            # Check thermal stability
            thermal_stability = stability.get('thermal_stability', 0)
            if thermal_stability < 0.5:
                strategies.append({
                    'type': 'stabilizing_mutations',
                    'description': f'Low thermal stability ({thermal_stability:.2f}). Consider stabilizing mutations in framework regions.',
                    'priority': 'medium'
                })
        
        # General strategies
        strategies.append({
            'type': 'framework_optimization',
            'description': 'Consider optimizing framework regions for improved stability while preserving antigen binding.',
            'priority': 'medium'
        })
        
        strategies.append({
            'type': 'glycosylation_engineering',
            'description': 'Consider engineering glycosylation sites to improve stability and effector functions.',
            'priority': 'low'
        })
        
        return strategies
    
    def _recommend_design_improvements(self, developability_predictions):
        """
        Recommend design improvements based on developability predictions.
        
        Args:
            developability_predictions (dict): Developability predictions
            
        Returns:
            list: Design improvement recommendations
        """
        improvements = []
        
        # Check solubility
        solubility = developability_predictions.get('solubility', 0.5)
        if solubility < 0.3:
            improvements.append({
                'type': 'solubility_improvement',
                'description': f'Low solubility ({solubility:.2f}). Consider surface engineering to increase solubility.',
                'priority': 'high'
            })
        
        # Check expression
        expression = developability_predictions.get('expression', 0.5)
        if expression < 0.3:
            improvements.append({
                'type': 'expression_optimization',
                'description': f'Low expression ({expression:.2f}). Consider codon optimization and framework stabilization.',
                'priority': 'high'
            })
        
        # Check aggregation
        aggregation = developability_predictions.get('aggregation', 0.5)
        if aggregation > 0.7:
            improvements.append({
                'type': 'aggregation_reduction',
                'description': f'High aggregation propensity ({aggregation:.2f}). Consider sequence modifications to reduce aggregation.',
                'priority': 'high'
            })
        
        # Check immunogenicity
        immunogenicity = developability_predictions.get('immunogenicity', 0.5)
        if immunogenicity > 0.7:
            improvements.append({
                'type': 'immunogenicity_reduction',
                'description': f'High immunogenicity ({immunogenicity:.2f}). Consider humanization or sequence modifications to reduce immunogenicity.',
                'priority': 'high'
            })
        
        # Overall developability
        overall_score = developability_predictions.get('overall_score', 0.5)
        if overall_score < 0.3:
            improvements.append({
                'type': 'comprehensive_optimization',
                'description': f'Low overall developability ({overall_score:.2f}). Consider comprehensive optimization approach.',
                'priority': 'high'
            })
        
        return improvements

if __name__ == "__main__":
    # Basic test of the OptimizationRecommender class
    print("Testing OptimizationRecommender:")
    
    # Create a recommender instance
    recommender = OptimizationRecommender()
    
    # Create sample fragment analysis data
    sample_analysis = {
        'composition': {
            'total_amino_acids': 117,
            'groups': {
                'hydrophobic': 65,
                'charged': 20,
                'polar': 32,
                'aromatic': 18
            },
            'counts': {
                'C': 2
            }
        },
        'physicochemical_properties': {
            'hydrophobicity': -0.2632,
            'isoelectric_point': 7.88
        },
        'stability': {
            'aggregation_propensity': 0.5744,
            'thermal_stability': 0.5767
        }
    }
    
    # Create sample developability predictions
    sample_predictions = {
        'solubility': 0.4293,
        'expression': 0.3262,
        'aggregation': 0.5744,
        'immunogenicity': 0.5169,
        'overall_score': 0.0419
    }
    
    # Generate recommendations
    recommendations = recommender.generate_recommendations(sample_analysis, sample_predictions)
    
    print(f"Priority: {recommendations['priority']}")
    print(f"Sequence modifications: {len(recommendations['sequence_modifications'])}")
    print(f"Structural strategies: {len(recommendations['structural_strategies'])}")
    print(f"Design improvements: {len(recommendations['design_improvements'])}")
    
    # Print detailed recommendations
    print("\nDetailed recommendations:")
    
    print("\nSequence modifications:")
    for i, mod in enumerate(recommendations['sequence_modifications']):
        print(f"  {i+1}. {mod['type']}: {mod['description']} (Priority: {mod['priority']})")
    
    print("\nStructural strategies:")
    for i, strategy in enumerate(recommendations['structural_strategies']):
        print(f"  {i+1}. {strategy['type']}: {strategy['description']} (Priority: {strategy['priority']})")
    
    print("\nDesign improvements:")
    for i, improvement in enumerate(recommendations['design_improvements']):
        print(f"  {i+1}. {improvement['type']}: {improvement['description']} (Priority: {improvement['priority']})")
