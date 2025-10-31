import re

# Read the file
with open('/a0/bitcore/workspace/flab_framework/developability_predictor_ensemble/developability_predictor_ensemble.py', 'r') as f:
    content = f.read()

# Define the new method
new_method = '''    def _calculate_overall_developability(self, solubility: float, expression: float, 
                                         aggregation: float, immunogenicity: float) -> float:
        """
        Calculate overall developability score.
        
        Args:
            solubility (float): Solubility score
            expression (float): Expression level score
            aggregation (float): Aggregation propensity score
            immunogenicity (float): Immunogenicity score
            
        Returns:
            float: Overall developability score (0-1)
        """
        # Calculate weighted sum
        score = (
            solubility * self.developability_weights['solubility'] +
            expression * self.developability_weights['expression'] +
            aggregation * self.developability_weights['aggregation'] +
            immunogenicity * self.developability_weights['immunogenicity']
        )
        
        # Calculate theoretical min and max scores based on weights
        # For scores between 0 and 1, min score occurs when positive weights are 0 and negative weights are 1
        # Max score occurs when positive weights are 1 and negative weights are 0
        min_score = (
            0 * self.developability_weights['solubility'] +
            0 * self.developability_weights['expression'] +
            1 * self.developability_weights['aggregation'] +
            1 * self.developability_weights['immunogenicity']
        )
        
        max_score = (
            1 * self.developability_weights['solubility'] +
            1 * self.developability_weights['expression'] +
            0 * self.developability_weights['aggregation'] +
            0 * self.developability_weights['immunogenicity']
        )
        
        # Normalize to 0-1 range
        if max_score != min_score:
            normalized_score = (score - min_score) / (max_score - min_score)
        else:
            normalized_score = 0.5  # Default if all weights are equal
        
        # Clamp score between 0 and 1
        normalized_score = max(0.0, min(1.0, normalized_score))
        
        return normalized_score'''

# Replace the old method with the new one
pattern = r'def _calculate_overall_developability\([^}]+return score\s*}'
content = re.sub(pattern, new_method, content, flags=re.DOTALL)

# Write the updated content back to the file
with open('/a0/bitcore/workspace/flab_framework/developability_predictor_ensemble/developability_predictor_ensemble.py', 'w') as f:
    f.write(content)
