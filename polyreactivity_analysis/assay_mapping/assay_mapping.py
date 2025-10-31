"""
PSR/PSP Assay Mapping Implementation
"""

class AssayMapper:
    """
    Implementation of PSR/PSP Assay Mapping
    """
    
    def __init__(self):
        """
        Initialize Assay Mapper
        """
        pass
    
    def calculate_psr(self, charge_imbalance, clustering_risk, binding_potential, dynamics_risk):
        """
        Calculate Polyreactivity Specificity Ratio (PSR)
        
        Parameters:
        charge_imbalance (float): Charge imbalance score
        clustering_risk (float): Clustering risk score
        binding_potential (float): Binding potential score
        dynamics_risk (float): Dynamics risk score
        
        Returns:
        float: PSR score (0-1, lower is better)
        """
        # Formula from documentation: 0.3 * charge_imbalance + 0.3 * clustering_risk + 0.2 * binding_potential + 0.2 * dynamics_risk
        psr = (
            0.3 * charge_imbalance +
            0.3 * clustering_risk +
            0.2 * binding_potential +
            0.2 * dynamics_risk
        )
        
        return psr
    
    def calculate_psp(self, psr):
        """
        Calculate Polyreactivity Specificity Potential (PSP)
        
        Parameters:
        psr (float): Polyreactivity Specificity Ratio score
        
        Returns:
        float: PSP score (0-1, lower is better)
        """
        # Calculate "improvability factor" (0-1, higher means easier to improve)
        improvability = 1.0 - psr  # Inverse relationship
        
        # PSP is PSR adjusted by improvability
        # Formula from documentation: psp = psr * (1.0 + 0.5 * (1.0 - improvability))
        psp = psr * (1.0 + 0.5 * (1.0 - improvability))
        
        return psp
    
    def assess_polyreactivity(self, psr):
        """
        Assess polyreactivity based on PSR score
        
        Parameters:
        psr (float): Polyreactivity Specificity Ratio score
        
        Returns:
        str: Polyreactivity assessment
        """
        if psr < 0.2:
            return "Low polyreactivity - favorable for specificity"
        elif psr < 0.4:
            return "Moderate polyreactivity - generally acceptable"
        elif psr < 0.6:
            return "High polyreactivity - may affect specificity"
        else:
            return "Very high polyreactivity - likely to cause specificity issues"
    
    def assess_developability_risk(self, psp):
        """
        Assess developability risk based on PSP score
        
        Parameters:
        psp (float): Polyreactivity Specificity Potential score
        
        Returns:
        str: Developability risk assessment
        """
        if psp < 0.3:
            return "Low developability risk"
        elif psp < 0.5:
            return "Moderate developability risk"
        elif psp < 0.7:
            return "High developability risk"
        else:
            return "Very high developability risk"
    
    def recommend_actions(self, charge_imbalance, clustering_risk, binding_potential, dynamics_risk):
        """
        Recommend actions based on individual feature scores
        
        Parameters:
        charge_imbalance (float): Charge imbalance score
        clustering_risk (float): Clustering risk score
        binding_potential (float): Binding potential score
        dynamics_risk (float): Dynamics risk score
        
        Returns:
        list: List of recommended actions
        """
        recommendations = []
        
        # Based on decision rules from documentation
        if charge_imbalance > 0.3:
            recommendations.append("Consider charge balancing modifications")
        
        if clustering_risk > 0.3:
            recommendations.append("Consider breaking up residue clusters")
        
        if binding_potential > 0.4:
            recommendations.append("Consider reducing hydrophobic patches")
        
        if dynamics_risk > 0.4:
            recommendations.append("Consider stabilizing paratope dynamics")
        
        return recommendations
    
    def determine_priority_ranking(self, psr, psp):
        """
        Determine priority ranking based on PSR and PSP scores
        
        Parameters:
        psr (float): Polyreactivity Specificity Ratio score
        psp (float): Polyreactivity Specificity Potential score
        
        Returns:
        str: Priority ranking
        """
        # Based on decision rules from documentation
        if psr > 0.6 or psp > 0.7:
            return "High - immediate attention required"
        elif psr > 0.4 or psp > 0.5:
            return "Medium - should be addressed"
        else:
            return "Low - monitor but no immediate action needed"
    
    def map_polyreactivity_features(self, sequence, charge_imbalance_score, clustering_risk_score, binding_potential, dynamics_risk_score):
        """
        Map polyreactivity features to assay data and create decision rules
        
        Parameters:
        sequence (str): Antibody sequence
        charge_imbalance_score (float): Charge imbalance score
        clustering_risk_score (float): Clustering risk score
        binding_potential (float): Binding potential score
        dynamics_risk_score (float): Dynamics risk score
        
        Returns:
        dict: Assay mapping results
        """
        # Calculate PSR
        psr = self.calculate_psr(charge_imbalance_score, clustering_risk_score, binding_potential, dynamics_risk_score)
        
        # Calculate PSP
        psp = self.calculate_psp(psr)
        
        # Assess polyreactivity
        polyreactivity_assessment = self.assess_polyreactivity(psr)
        
        # Assess developability risk
        developability_risk = self.assess_developability_risk(psp)
        
        # Recommend actions
        recommendations = self.recommend_actions(charge_imbalance_score, clustering_risk_score, binding_potential, dynamics_risk_score)
        
        # Determine priority ranking
        priority_ranking = self.determine_priority_ranking(psr, psp)
        
        return {
            'psr': psr,
            'psp': psp,
            'polyreactivity_assessment': polyreactivity_assessment,
            'developability_risk': developability_risk,
            'recommendations': recommendations,
            'priority_ranking': priority_ranking
        }
    
    def generate_assay_report(self, sequence, charge_imbalance_score, clustering_risk_score, binding_potential, dynamics_risk_score):
        """
        Generate a comprehensive assay report
        
        Parameters:
        sequence (str): Antibody sequence
        charge_imbalance_score (float): Charge imbalance score
        clustering_risk_score (float): Clustering risk score
        binding_potential (float): Binding potential score
        dynamics_risk_score (float): Dynamics risk score
        
        Returns:
        dict: Assay report
        """
        # Map polyreactivity features
        assay_results = self.map_polyreactivity_features(
            sequence, charge_imbalance_score, clustering_risk_score, binding_potential, dynamics_risk_score
        )
        
        # Generate summary
        summary = f"Assay Report\n"
        summary += f"PSR: {assay_results['psr']:.4f}\n"
        summary += f"PSP: {assay_results['psp']:.4f}\n"
        summary += f"Polyreactivity Assessment: {assay_results['polyreactivity_assessment']}\n"
        summary += f"Developability Risk: {assay_results['developability_risk']}\n"
        summary += f"Priority Ranking: {assay_results['priority_ranking']}\n"
        summary += f"Recommendations: {', '.join(assay_results['recommendations']) if assay_results['recommendations'] else 'None'}\n"
        
        return {
            'summary': summary,
            'results': assay_results
        }

def main():
    """
    Main function for testing Assay Mapper
    """
    print("Testing Assay Mapper...")
    
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
    assay_results = mapper.map_polyreactivity_features(
        sequence, charge_imbalance_score, clustering_risk_score, binding_potential, dynamics_risk_score
    )
    print(f"PSR: {assay_results['psr']:.3f}")
    print(f"PSP: {assay_results['psp']:.3f}")
    print(f"Polyreactivity Assessment: {assay_results['polyreactivity_assessment']}")
    print(f"Developability Risk: {assay_results['developability_risk']}")
    print(f"Recommendations: {assay_results['recommendations']}")
    print(f"Priority Ranking: {assay_results['priority_ranking']}")
    
    # Generate comprehensive assay report
    assay_report = mapper.generate_assay_report(
        sequence, charge_imbalance_score, clustering_risk_score, binding_potential, dynamics_risk_score
    )
    print("\nAssay Report:")
    print(assay_report['summary'])
    
    print("\nAssay Mapper test completed successfully!")


if __name__ == '__main__':
    main()
