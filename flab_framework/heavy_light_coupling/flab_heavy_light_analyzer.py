"""
FLAb Framework Integration for Enhanced Heavy-Light Coupling Analysis with Subclass-Specific Developability Prediction

This module provides FLAb framework integration for the enhanced heavy-light coupling analysis.
"""

import sys
import os

# Add path to research directory
sys.path.insert(0, '/a0/bitcore/workspace/research/heavy_light_coupling/src')

# Import the enhanced heavy-light analyzer
from enhanced_heavy_light_analyzer_with_subclass import EnhancedHeavyLightAnalyzer

class FLAbHeavyLightAnalyzer:
    """
    FLAb framework integration for enhanced heavy-light coupling analysis with subclass-specific developability prediction.
    """

    def __init__(self):
        """
        Initialize the FLAb heavy-light analyzer.
        """
        self.analyzer = EnhancedHeavyLightAnalyzer()

    def analyze_antibody(self, vh_sequence: str, vl_sequence: str,
                        heavy_chain_sequence: str = None, light_chain_sequence: str = None,
                        isotype: str = 'IgG1') -> dict:
        """
        Analyze an antibody's heavy-light coupling with subclass-specific developability prediction.

        Args:
            vh_sequence (str): VH chain amino acid sequence
            vl_sequence (str): VL chain amino acid sequence
            heavy_chain_sequence (str): Full heavy chain amino acid sequence (optional)
            light_chain_sequence (str): Full light chain amino acid sequence (optional)
            isotype (str): Antibody isotype

        Returns:
            dict: Heavy-light coupling analysis results with subclass-specific developability prediction
        """
        # If full chain sequences are not provided, use VH/VL sequences
        if heavy_chain_sequence is None:
            heavy_chain_sequence = vh_sequence

        if light_chain_sequence is None:
            light_chain_sequence = vl_sequence

        # Generate coupling report
        coupling_report = self.analyzer.generate_coupling_report(
            vh_sequence, vl_sequence, heavy_chain_sequence, light_chain_sequence, isotype
        )

        return coupling_report

    def get_pairing_score(self, vh_sequence: str, vl_sequence: str) -> float:
        """
        Get the VH-VL pairing score.

        Args:
            vh_sequence (str): VH chain amino acid sequence
            vl_sequence (str): VL chain amino acid sequence

        Returns:
            float: Pairing score (0-1)
        """
        pairing_analysis = self.analyzer.analyze_vh_vl_pairing(vh_sequence, vl_sequence)
        return pairing_analysis['pairing_score']

    def get_isotype_score(self, heavy_chain_sequence: str, light_chain_sequence: str, isotype: str = 'IgG1') -> float:
        """
        Get the isotype score.

        Args:
            heavy_chain_sequence (str): Full heavy chain amino acid sequence
            light_chain_sequence (str): Full light chain amino acid sequence
            isotype (str): Antibody isotype

        Returns:
            float: Isotype score (0-1)
        """
        isotype_analysis = self.analyzer.analyze_isotype_features(heavy_chain_sequence, light_chain_sequence, isotype)
        return isotype_analysis['isotype_score']

    def get_subclass_developability(self, isotype: str = 'IgG1') -> dict:
        """
        Get subclass-specific developability predictions.

        Args:
            isotype (str): Antibody isotype

        Returns:
            dict: Predicted developability properties for the subclass
        """
        return self.analyzer.predict_subclass_developability(isotype)

    def get_overall_compatibility(self, vh_sequence: str, vl_sequence: str,
                                heavy_chain_sequence: str = None, light_chain_sequence: str = None,
                                isotype: str = 'IgG1') -> float:
        """
        Get the overall heavy-light compatibility score.

        Args:
            vh_sequence (str): VH chain amino acid sequence
            vl_sequence (str): VL chain amino acid sequence
            heavy_chain_sequence (str): Full heavy chain amino acid sequence (optional)
            light_chain_sequence (str): Full light chain amino acid sequence (optional)
            isotype (str): Antibody isotype

        Returns:
            float: Overall compatibility score (0-1)
        """
        # If full chain sequences are not provided, use VH/VL sequences
        if heavy_chain_sequence is None:
            heavy_chain_sequence = vh_sequence

        if light_chain_sequence is None:
            light_chain_sequence = vl_sequence

        coupling_report = self.analyzer.generate_coupling_report(
            vh_sequence, vl_sequence, heavy_chain_sequence, light_chain_sequence, isotype
        )

        return coupling_report['overall_compatibility']


def main():
    """
    Example usage of the FLAb heavy-light analyzer.
    """
    # Example sequences
    vh_sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    vl_sequence = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLAWYQQKPGKAPKLLIYDASNLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSNHWPTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
    heavy_chain_sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSREEMTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK"
    light_chain_sequence = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLAWYQQKPGKAPKLLIYDASNLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSNHWPTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"

    # Create FLAb analyzer
    flab_analyzer = FLAbHeavyLightAnalyzer()

    # Analyze antibody
    coupling_report = flab_analyzer.analyze_antibody(vh_sequence, vl_sequence, heavy_chain_sequence, light_chain_sequence, 'IgG1')
    print("FLAb Heavy-Light Coupling Analysis:")
    print(coupling_report['summary'])

    # Get pairing score
    pairing_score = flab_analyzer.get_pairing_score(vh_sequence, vl_sequence)
    print(f"\nPairing Score: {pairing_score:.3f}")

    # Get isotype score
    isotype_score = flab_analyzer.get_isotype_score(heavy_chain_sequence, light_chain_sequence, 'IgG1')
    print(f"Isotype Score: {isotype_score:.3f}")

    # Get subclass-specific developability
    subclass_developability = flab_analyzer.get_subclass_developability('IgG1')
    print("\nSubclass-Specific Developability Prediction:")
    for prop, value in subclass_developability.items():
        print(f"  {prop}: {value:.3f}")

    # Get overall compatibility
    overall_compatibility = flab_analyzer.get_overall_compatibility(vh_sequence, vl_sequence, heavy_chain_sequence, light_chain_sequence, 'IgG1')
    print(f"\nOverall Compatibility: {overall_compatibility:.3f}")


if __name__ == "__main__":
    main()
