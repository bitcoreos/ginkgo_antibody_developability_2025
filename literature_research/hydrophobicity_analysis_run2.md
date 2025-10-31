# Analysis Brief: IgFold "Structure-Lite" Contact Proxies vs. Observed HIC Variance {#analysis-brief-igfold-structure-lite-contact-proxies-vs.-observed-hic-variance .unnumbered}

## Scope {#scope .unnumbered}

Evaluate whether fast, backbone-focused antibody models (IgFold) yield
surface/contacts proxies that explain variance in Hydrophobic
Interaction Chromatography (HIC) readouts; identify limits and controls.

## Assay & Mechanism Facts {#assay-mechanism-facts .unnumbered}

HIC retention primarily reflects desolvation and adsorption of exposed
hydrophobic surface patches under high salt; retention is modulated by
salt identity/strength, pH relative to pI, temperature, gradient shape,
ligand chemistry, and load. Hydrophobic patches dominate signal;
electrostatics modulate it near pI. *Implication*: features must capture
patch size/topology and local charge context. (complete cites in
Traceability)

## Model Substrate {#model-substrate .unnumbered}

**IgFold** predicts antibody Fv structures from sequence in seconds with
per-residue error estimates; outputs are refined to full-atom models but
remain "structure-lite" versus MD ensembles. *Implication*: adequate for
coarse patch geometry; limited for conformational averaging and
side-chain microstates.

## Feature Set: Structure-Lite Contact Proxies {#feature-set-structure-lite-contact-proxies .unnumbered}

Given an IgFold Fv:

1.  **Nonpolar SASA** (NP-SASA): solvent-exposed ASA of
    aliphatic/aromatic side chains; report total and by domain (VH/VL)
    and by regions (CDRs vs FR).

2.  **Largest Contiguous Hydrophobic Patch** (LCHP): maximum surface
    patch size (Å$^{2}$) of connected nonpolar vertices on a
    solvent-excluded mesh; record patch count and Gini of patch areas.

3.  **Hydrophobic Contact Density** (HCD): count of side-chain
    heavy-atom contacts within 4.5--5.0 Å where both residues are
    hydrophobic; stratify by surface accessibility to penalize buried
    contacts.

4.  **Aromatic Exposure Index** (AEI): sum of exposed ring-centroid SASA
    for F, W, Y; separate for CDR-H3 tip vs. scaffold.

5.  **Patch Charge Context** (PCC): net Coulombic potential from
    K/R/H/D/E neighbors within 10 Å of the LCHP centroid; captures
    charge--hydrophobe coupling that affects retention near pI.

6.  **Patch Hydropathy Gradient** (PHG): local Kyte--Doolittle or Jain
    HIC-trained scale mean within 6 Å vs. bulk surface mean; proxy for
    patch contrast.

7.  **Uncertainty Weights** (UW): per-residue IgFold error used to
    down-weight contributions; compute feature means across $N$
    side-chain rotamer samplings to approximate microstate variance.

## Statistical Plan {#statistical-plan .unnumbered}

1.  **Targets**: HIC retention time or normalized RT; optionally
    "delayed retention" classification.

2.  **Associations**: Spearman $\rho$ between each proxy and HIC; report
    95% CIs via bootstrap; partial correlations controlling for net
    charge/pI.

3.  **Models**: Elastic net and gradient boosting on standardized
    features {[NP-SASA, LCHP, HCD, AEI, PCC, PHG]{.sans-serif}} +
    sequence covariates (length, pI, net charge).

4.  **Controls**: Column ligand type, salt, pH, gradient slope as
    categorical/continuous covariates if available; batch and date as
    random effects.

5.  **Uncertainty propagation**: Monte Carlo over rotamers and IgFold
    error masks; report variance explained $R^{2}$ distributions.

## Expected Signal {#expected-signal .unnumbered}

**Literature-anchored priors**:

-   Structure-based surface patch descriptors (AggScore/MOE patches,
    SAP-like metrics) correlate with HIC and aggregation; sequence-only
    hydropathy is weaker.

-   Increased structural fidelity (MD-relaxed ensembles) improves HIC
    prediction over homology models; thus IgFold likely performs between
    sequence-only and MD.

-   Conformational ensembles influence apparent hydrophobicity; single
    static structures can under- or over-estimate patch sizes on
    flexible CDRs, especially H3.

**Inference**: IgFold proxies should capture *directional* HIC risk
(rank-ordering) but will under-explain variance when patch expression is
conformation-dependent or when buffer/column effects dominate.

## Variance Sources to Account For {#variance-sources-to-account-for .unnumbered}

1.  **Assay**: salt type/strength, pH vs. pI, temperature, gradient
    program, stationary phase ligand density/chemistry, load, system
    salt memory and column aging.

2.  **Analyte**: local positive patches adjacent to hydrophobes, Met/Trp
    oxidation state, glycan heterogeneity, ADC payloads (if any),
    post-translational states.

3.  **Modeling**: homology/template bias, side-chain placement, lack of
    dynamics; different hydrophobicity scales produce materially
    different HIC predictions.

## Decision Rules (practical) {#decision-rules-practical .unnumbered}

-   Flag as HIC-risk if any of: top-decile LCHP, top-decile AEI in
    CDR-H3, PCC indicates locally positive environment, or NP-SASA $>$
    threshold calibrated on a reference panel.

-   For borderline cases, require ensemble stability: proxies must
    remain above thresholds across uncertainty samplings.

## Limitations {#limitations .unnumbered}

Structure-lite proxies lack explicit solvent and conformational
averaging; charge--hydrophobe coupling is pH-dependent; assay-to-assay
differences can exceed feature differences; generalization across column
chemistries is non-trivial.

## Next Validation {#next-validation .unnumbered}

Reproduce published HIC panels; benchmark: (i) sequence-only
hydropathy+exposure estimates, (ii) IgFold proxies (this brief), (iii)
MD-relaxed ensembles. Compare AUROC for delayed-retention classification
and $R^{2}_{\mathrm{ext}}$ for RT prediction; report calibration curves.
Use per-study salt/pH/column covariates where disclosed.

## Key Takeaway {#key-takeaway .unnumbered}

Use IgFold to get fast, uncertainty-aware hydrophobic patch proxies that
rank HIC risk and explain a *portion* of variance; expect improved but
sub-MD performance; always co-model charge and assay conditions;
validate per-laboratory protocol.
