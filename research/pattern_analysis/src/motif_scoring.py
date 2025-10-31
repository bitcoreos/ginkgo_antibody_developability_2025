"""
Motif-Based Risk Scoring Module

This module implements motif-based risk scoring beyond current implementations.
"""

import numpy as np
from typing import Dict, List, Tuple, Union
from collections import Counter
import json
import os

# Motif database file path
MOTIF_DATABASE_FILE = "/a0/bitcore/workspace/research/pattern_analysis/data/motif_database.json"

# Default motif database
DEFAULT_MOTIF_DATABASE = {
    "aggregation_prone": {
        "motifs": [
            {"sequence": "GGG", "risk_score": 0.8, "description": "Glycine-rich regions associated with aggregation"},
            {"sequence": "WWW", "risk_score": 0.8, "description": "Tryptophan-rich regions associated with aggregation"},
            {"sequence": "FFFF", "risk_score": 0.8, "description": "Phenylalanine-rich regions associated with aggregation"},
            {"sequence": "YYYY", "risk_score": 0.7, "description": "Tyrosine-rich regions associated with aggregation"},
            {"sequence": "MMM", "risk_score": 0.7, "description": "Methionine-rich regions susceptible to oxidation"},
            {"sequence": "AAAA", "risk_score": 0.6, "description": "Alanine-rich regions associated with aggregation"},
            {"sequence": "VVVV", "risk_score": 0.6, "description": "Valine-rich regions associated with aggregation"},
            {"sequence": "IIII", "risk_score": 0.6, "description": "Isoleucine-rich regions associated with aggregation"},
            {"sequence": "LLLL", "risk_score": 0.6, "description": "Leucine-rich regions associated with aggregation"}
        ]
    },
    "stability_issues": {
        "motifs": [
            {"sequence": "CC", "risk_score": 0.7, "description": "Cysteine pairs that may form incorrect disulfide bonds"},
            {"sequence": "DD", "risk_score": 0.6, "description": "Aspartic acid pairs that may cause isomerization"},
            {"sequence": "NN", "risk_score": 0.6, "description": "Asparagine pairs that may cause deamidation"},
            {"sequence": "PP", "risk_score": 0.5, "description": "Proline pairs that may affect folding"},
            {"sequence": "SS", "risk_score": 0.4, "description": "Serine pairs that may cause unwanted interactions"},
            {"sequence": "TT", "risk_score": 0.4, "description": "Threonine pairs that may cause unwanted interactions"}
        ]
    },
    "cleavage_sites": {
        "motifs": [
            {"sequence": "FR", "risk_score": 0.6, "description": "Phe-Arg motifs associated with proteolytic cleavage"},
            {"sequence": "KR", "risk_score": 0.5, "description": "Lys-Arg motifs associated with proteolytic cleavage"},
            {"sequence": "RR", "risk_score": 0.5, "description": "Arg-Arg motifs associated with proteolytic cleavage"}
        ]
    },
    "deamidation_sites": {
        "motifs": [
            {"sequence": "NG", "risk_score": 0.6, "description": "Asn-Gly motifs associated with deamidation"},
            {"sequence": "NS", "risk_score": 0.5, "description": "Asn-Ser motifs associated with deamidation"},
            {"sequence": "NT", "risk_score": 0.5, "description": "Asn-Thr motifs associated with deamidation"}
        ]
    },
    "isomerization_sites": {
        "motifs": [
            {"sequence": "DG", "risk_score": 0.6, "description": "Asp-Gly motifs associated with isomerization"},
            {"sequence": "DS", "risk_score": 0.5, "description": "Asp-Ser motifs associated with isomerization"},
            {"sequence": "DT", "risk_score": 0.5, "description": "Asp-Thr motifs associated with isomerization"}
        ]
    }
}


class MotifScorer:
    """
    Scorer for motif-based risk assessment in antibody sequences.
    """
    
    def __init__(self, database_file: str = MOTIF_DATABASE_FILE):
        """
        Initialize the motif scorer.
        
        Args:
            database_file (str): Path to motif database file
        """
        self.database_file = database_file
        self.motif_database = self._load_motif_database()
    
    def _load_motif_database(self) -> Dict:
        """
        Load motif database from file or create default database.
        
        Returns:
            Dict: Motif database
        """
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.database_file), exist_ok=True)
        
        # If database file exists, load it
        if os.path.exists(self.database_file):
            try:
                with open(self.database_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading motif database: {e}")
                return DEFAULT_MOTIF_DATABASE
        else:
            # Create default database file
            with open(self.database_file, 'w') as f:
                json.dump(DEFAULT_MOTIF_DATABASE, f, indent=2)
            return DEFAULT_MOTIF_DATABASE
    
    def _save_motif_database(self):
        """
        Save motif database to file.
        """
        try:
            with open(self.database_file, 'w') as f:
                json.dump(self.motif_database, f, indent=2)
        except Exception as e:
            print(f"Error saving motif database: {e}")
    
    def score_motifs(self, sequence: str) -> Dict[str, Union[int, float, List[Dict]]]:
        """
        Score motifs in an antibody sequence.
        
        Args:
            sequence (str): Amino acid sequence
            
        Returns:
            Dict: Motif scoring results
        """
        sequence = sequence.upper()
        length = len(sequence)
        
        if length == 0:
            return {
                'sequence': sequence,
                'length': 0,
                'motifs': [],
                'motif_count': 0,
                'total_risk_score': 0.0,
                'category_scores': {},
                'scoring_complete': True
            }
        
        # Score motifs from all categories
        identified_motifs = []
        category_scores = {}
        
        for category, category_data in self.motif_database.items():
            category_motifs = category_data.get('motifs', [])
            category_motif_scores = []
            
            for motif_data in category_motifs:
                motif_seq = motif_data['sequence']
                base_risk_score = motif_data['risk_score']
                description = motif_data['description']
                
                # Count occurrences of motif
                count = sequence.count(motif_seq)
                if count > 0:
                    # Calculate adjusted risk score based on count
                    # Using a diminishing returns model where additional occurrences
                    # contribute less to the overall risk
                    adjusted_risk_score = 1 - (1 - base_risk_score) ** count
                    
                    identified_motifs.append({
                        'category': category,
                        'motif': motif_seq,
                        'count': count,
                        'positions': self._find_motif_positions(sequence, motif_seq),
                        'base_risk_score': base_risk_score,
                        'adjusted_risk_score': adjusted_risk_score,
                        'description': description
                    })
                    
                    # Add to category scores
                    category_motif_scores.append(adjusted_risk_score)
            
            # Calculate category score as maximum of motif scores in category
            if category_motif_scores:
                category_scores[category] = max(category_motif_scores)
            else:
                category_scores[category] = 0.0
        
        # Calculate total risk score as weighted average of category scores
        total_risk_score = self._calculate_total_risk_score(category_scores)
        
        return {
            'sequence': sequence,
            'length': length,
            'motifs': identified_motifs,
            'motif_count': len(identified_motifs),
            'total_risk_score': total_risk_score,
            'category_scores': category_scores,
            'scoring_complete': True
        }
    
    def _find_motif_positions(self, sequence: str, motif: str) -> List[int]:
        """
        Find all positions of a motif in a sequence.
        
        Args:
            sequence (str): Amino acid sequence
            motif (str): Motif to find
            
        Returns:
            List[int]: List of starting positions
        """
        positions = []
        start = 0
        while True:
            pos = sequence.find(motif, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        return positions
    
    def _calculate_total_risk_score(self, category_scores: Dict[str, float]) -> float:
        """
        Calculate total risk score from category scores.
        
        Args:
            category_scores (Dict[str, float]): Category scores
            
        Returns:
            float: Total risk score (0-1, higher is worse)
        """
        if not category_scores:
            return 0.0
        
        # Calculate weighted average of category scores
        # Using equal weights for all categories
        total_score = sum(category_scores.values())
        count = len(category_scores)
        
        if count == 0:
            return 0.0
        
        return min(1.0, total_score / count)
    
    def add_motif(self, category: str, motif: str, risk_score: float, description: str):
        """
        Add a new motif to the database.
        
        Args:
            category (str): Motif category
            motif (str): Motif sequence
            risk_score (float): Risk score (0-1)
            description (str): Motif description
        """
        # Ensure category exists
        if category not in self.motif_database:
            self.motif_database[category] = {'motifs': []}
        
        # Add motif to category
        self.motif_database[category]['motifs'].append({
            'sequence': motif,
            'risk_score': risk_score,
            'description': description
        })
        
        # Save updated database
        self._save_motif_database()
    
    def remove_motif(self, category: str, motif: str):
        """
        Remove a motif from the database.
        
        Args:
            category (str): Motif category
            motif (str): Motif sequence
        """
        # Check if category exists
        if category in self.motif_database:
            # Remove motif from category
            self.motif_database[category]['motifs'] = [
                m for m in self.motif_database[category]['motifs']
                if m['sequence'] != motif
            ]
            
            # Save updated database
            self._save_motif_database()
    
    def generate_motif_report(self, sequence: str) -> Dict[str, Union[str, float, List]]:
        """
        Generate a comprehensive motif scoring report.
        
        Args:
            sequence (str): Amino acid sequence
            
        Returns:
            Dict: Comprehensive motif scoring report
        """
        # Score motifs
        motif_results = self.score_motifs(sequence)
        
        # Extract key metrics
        motif_count = motif_results['motif_count']
        total_risk_score = motif_results['total_risk_score']
        category_scores = motif_results['category_scores']
        motifs = motif_results['motifs']
        
        # Generate risk assessment
        if total_risk_score < 0.2:
            risk_assessment = "Low risk - few or no problematic motifs identified"
        elif total_risk_score < 0.4:
            risk_assessment = "Moderate risk - some potentially problematic motifs identified"
        elif total_risk_score < 0.6:
            risk_assessment = "High risk - several problematic motifs identified"
        else:
            risk_assessment = "Very high risk - many problematic motifs identified"
        
        # Generate summary
        category_lines = [f"- {category}: {score:.3f}" for category, score in category_scores.items()]
        category_text = "\n".join(category_lines)
        summary = f"""
Motif Scoring Report
===================

Total Risk Score: {total_risk_score:.3f} ({risk_assessment})
Motif Count: {motif_count}

Category Scores:
{category_text}
"""
        
        return {
            'sequence': sequence,
            'total_risk_score': total_risk_score,
            'motif_count': motif_count,
            'category_scores': category_scores,
            'motifs': motifs,
            'risk_assessment': risk_assessment,
            'summary': summary,
            'report_complete': True
        }


def main():
    """
    Example usage of the motif scorer.
    """
    # Example sequence with various motifs
    sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    
    # Create scorer
    scorer = MotifScorer()
    
    # Score motifs
    motif_results = scorer.score_motifs(sequence)
    
    print("Motif Scoring Results:")
    print(f"  Sequence Length: {motif_results['length']}")
    print(f"  Motif Count: {motif_results['motif_count']}")
    print(f"  Total Risk Score: {motif_results['total_risk_score']:.3f}")
    
    # Print category scores
    print("\nCategory Scores:")
    for category, score in motif_results['category_scores'].items():
        print(f"  {category}: {score:.3f}")
    
    # Print identified motifs
    print("\nIdentified Motifs:")
    for i, motif in enumerate(motif_results['motifs']):
        print(f"  {i+1}. {motif['category']}: {motif['motif']}")
        print(f"     Count: {motif['count']}")
        print(f"     Base Risk Score: {motif['base_risk_score']:.3f}")
        print(f"     Adjusted Risk Score: {motif['adjusted_risk_score']:.3f}")
        print(f"     Description: {motif['description']}")
    
    # Generate comprehensive motif report
    motif_report = scorer.generate_motif_report(sequence)
    print("\nMotif Report Summary:")
    print(motif_report['summary'])
    
    # Example of adding a new motif
    print("\nAdding new motif...")
    scorer.add_motif("custom_category", "QQQ", 0.4, "Glutamine-rich regions")
    
    # Score motifs again with new motif
    motif_results_new = scorer.score_motifs(sequence)
    print(f"\nNew Total Risk Score: {motif_results_new['total_risk_score']:.3f}")


if __name__ == "__main__":
    main()
