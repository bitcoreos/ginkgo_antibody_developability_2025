# Polyreactivity Research Report for GDPa1 Dataset

## Executive Summary

This report analyzes polyreactivity (PR_CHO) signatures in the GDPa1 antibody dataset, focusing on three key aspects: VH/VL charge imbalance, basic residue patterns, and entropy-linked polyspecificity risks. Our analysis reveals significant relationships between these sequence-based features and polyreactivity measurements.

## 1. Threshold Mapping: VH/VL Charge Imbalance and PR_CHO Excursions

### Key Findings

1. **VH Net Charge**: Strong positive correlation with PR_CHO (Pearson r=0.301, p<0.001), indicating that antibodies with higher net charge in their heavy chain variable domains are more polyreactive.

2. **CDR Total Net Charge**: Significant positive correlation with PR_CHO (Pearson r=0.250, p<0.001).

3. **VL Net Charge**: Positive correlation with PR_CHO (Pearson r=0.191, p=0.007).

4. **CDR Negative Charge**: Negative correlation with PR_CHO (Pearson r=-0.203, p=0.004), suggesting that more negatively charged antibodies tend to have lower polyreactivity.

### Threshold Analysis

Comparing high PR_CHO (≥75th percentile, ≥0.335) versus low PR_CHO (<75th percentile, <0.335) groups:

- High PR_CHO antibodies have significantly higher VH net charge (2.23 vs 1.18)
- High PR_CHO antibodies have significantly higher VL net charge (1.86 vs 0.89)
- High PR_CHO antibodies show increased charge symmetry ratio (1.71 vs 1.37)

### Interpretation

These findings support the hypothesis that charge imbalance, particularly positive charge enrichment in VH and VL domains, contributes to polyreactivity against CHO lysate. The strong correlation between VH net charge and PR_CHO suggests that VH domain electrostatics are a primary driver of polyreactivity in this dataset.

## 2. Motif Atlas: Basic Residue Run-Length Patterns

### Key Findings

1. **CDR L1 Positive Charge**: Significant positive correlation with PR_CHO (Pearson r=0.189, p=0.008), indicating that antibodies with higher positive charge density in their light chain CDR1 are more polyreactive.

2. **CDR H3 Negative Charge**: Significant negative correlation with PR_CHO (Spearman r=-0.154, p=0.031), suggesting that more negatively charged CDR H3 regions are associated with lower polyreactivity.

3. **VH Positive Charge Clustering**: Strong negative correlation with PR_CHO (Pearson r=-0.271, p<0.001; Spearman r=-0.357, p<0.001), indicating that clustered positive charges in the VH domain are associated with lower polyreactivity.

4. **VL Positive Charge Clustering**: Positive correlation with PR_CHO (Spearman r=0.161, p=0.024).

### Motif Risk Patterns

Based on our analysis, the following patterns emerge as potential risk factors for polyreactivity:

- Enrichment of positive charges in CDR L1
- Lack of negative charge clustering in VH domains
- Dispersed (non-clustered) positive charges in VH domains
- Increased positive charge clustering in VL domains

### Interpretation

These findings suggest that the distribution and clustering of charged residues, particularly in the CDR regions, plays an important role in polyreactivity. The negative correlation between VH positive charge clustering and PR_CHO is particularly interesting, as it suggests that antibodies with more organized positive charge patches in their VH domains are less polyreactive.

## 3. Entropy Analysis: Polyspecificity Risk Stratification

### Key Findings

1. **CDR L3 Sequence Entropy**: Significant positive correlation with PR_CHO (Pearson r=0.197, p=0.006; Spearman r=0.184, p=0.010), indicating that higher sequence diversity in the light chain CDR3 is associated with higher polyreactivity.

2. **CDR H1 Sequence Entropy**: Significant negative correlation with PR_CHO (Pearson r=-0.164, p=0.022; Spearman r=-0.144, p=0.044), suggesting that higher sequence diversity in heavy chain CDR1 is associated with lower polyreactivity.

3. **VH Domain Entropy**: Strong negative correlations with PR_CHO:
   - vh_avg_local_entropy: Pearson r=-0.321, p<0.001
   - vh_sequence_entropy: Pearson r=-0.174, p=0.015
   - vh_entropy_variation: Pearson r=-0.212, p=0.003

### Entropy Risk Stratification

Our analysis reveals a complex relationship between entropy and polyreactivity:

- Higher entropy in CDR L3 is associated with increased polyreactivity
- Higher entropy in CDR H1 is associated with decreased polyreactivity
- Higher entropy in the VH domain is generally associated with decreased polyreactivity

### Interpretation

The entropy analysis suggests that sequence diversity has region-specific effects on polyreactivity. The strong negative correlation between VH domain entropy and PR_CHO is counterintuitive but statistically significant, possibly indicating that antibodies with more constrained VH sequences (lower entropy) are more likely to adopt conformations that promote polyreactivity.

## Conclusions

This analysis of the GDPa1 dataset reveals several key insights into polyreactivity determinants:

1. **Charge Imbalance**: VH net charge is the strongest single predictor of PR_CHO, with higher positive charge in VH domains strongly associated with increased polyreactivity.

2. **Basic Residue Patterns**: The distribution and clustering of charged residues, particularly in CDR regions, significantly impact polyreactivity. Clustered positive charges in VH domains are protective, while dispersed charges increase risk.

3. **Entropy Patterns**: Sequence diversity has complex, region-specific effects on polyreactivity. Higher entropy in CDR L3 increases polyreactivity risk, while higher entropy in VH domains appears protective.

## Recommendations

Based on these findings, we recommend the following for antibody engineering to reduce polyreactivity risk:

1. Minimize net positive charge in VH domains
2. Promote clustering of positive charges in VH domains rather than dispersing them
3. Reduce positive charge density in CDR L1
4. Consider the differential effects of entropy in different regions when designing antibodies

## Limitations

This analysis is based on correlations in a dataset of 197 antibodies with PR_CHO measurements. While the statistical associations are significant, causation cannot be definitively established without experimental validation. Additionally, the dataset represents a specific set of antibodies and may not be fully representative of all antibody sequences.

## References

1. Chen HT et al. Human antibody polyreactivity is governed primarily by the heavy chain. Cell Reports. 2024.
2. Raybould MIJ et al. TAP: A Web Server for the Rational Engineering of Therapeutic Antibodies. PNAS. 2019.
3. Lecerf M et al. Polyreactivity of antibodies from different B-cell populations. Front Immunol. 2023.
4. Cunningham O et al. Polyreactivity and polyspecificity in therapeutic antibody development. mAbs. 2021.
5. Makowski EK et al. Highly sensitive detection of antibody nonspecific interactions using flow cytometry. mAbs. 2021.
