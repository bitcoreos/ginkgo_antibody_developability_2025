# Ultimate Advantage: Isotype-Specific Structural Motifs for Tm2 Prediction

## Context
Current models use categorical isotype encoding (IgG1, IgG2, IgG4) for Tm2 prediction, but miss the underlying biophysical mechanism. IgG1 Fab domains are inherently more stable than IgG4 due to stronger CH1-CL interface interactions, a structural difference not captured by categorical labels.

## Reasoning
By modeling the CH1-CL interface stability as a continuous feature (hydrophobicity + H-bonds), we replace coarse categorization with a biophysically accurate, granular representation. This first-principles approach aligns with the physical reality of antibody stability, enabling more precise predictions and better generalization to novel isotypes.

## Evidence
- **PMC3631360**: Direct biophysical comparison shows IgG1 Fabs have higher Tm2 than IgG4 Fabs due to CH1-CL interface differences.
- **Structural Analysis**: IgG1 has more hydrophobic interface residues and additional hydrogen bonds vs. IgG4.
- **Memory ID hZ9By5ysOC**: Strategic memory validates this as a novel insight beyond existing isotype stratification.

## Confidence
0.8 — Strong experimental evidence from direct comparison. Justification: Study uses identical variable regions, isolating isotype effect. Limited by n=14, but consistent with structural principles.

## Implementation Path
1. **Extract Domains**: Parse CH1 and CL sequences from full antibody.
2. **Calculate Score**: 
   - Interface residues: [known list]
   - Hydrophobicity: Sum Kyte-Doolittle scores of interface residues.
   - H-bonds: Count donor-acceptor pairs (e.g., Arg-Asp, Lys-Glu).
   - Final Score = 0.7*Hydrophobicity + 0.3*H-bonds.
3. **Model Integration**: Add score as continuous feature alongside isotype category.
4. **Validation**: Ablation study — compare model with categorical vs. structural feature.
5. **Expected Gain**: +0.05 Spearman on Tm2 leaderboard.
