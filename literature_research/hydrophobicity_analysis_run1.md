"'latex

# Analysis brief: "structure-lite" (IgFold) contact proxies vs observed HIC variance {#analysis-brief-structure-lite-igfold-contact-proxies-vs-observed-hic-variance .unnumbered}

**Scope.** Map 3D-lite features from IgFold-predicted Fv structures to
HIC retention variability; identify signal-bearing proxies, confounders,
and validation steps.

**Definition.** "Structure-lite" = single IgFold static model of VH/VL
(no MD, no glycan, no explicit resin model, no conformational ensemble).

**Mechanism premise.** HIC retention increases with size and exposure of
hydrophobic surface patches (incl. aromatics) under high-salt charge
screening; charge distribution and resin chemistry modulate signal. "'

(\[PMC\]\[1\])

"'latex

# Contact-proxy feature set from IgFold {#contact-proxy-feature-set-from-igfold .unnumbered}

1.  **Hydrophobic SASA**: sum and fraction of SASA for
    {A,V,I,L,M,F,W,Y}; CDR-specific and global.

2.  **Largest hydrophobic patch area**: size of maximal connected
    surface patch (graph on exposed hydrophobes).

3.  **Aromatic exposure**: SASA and patch density for F/W/Y; CDR-H3
    emphasis.

4.  **Hydrophobic contact density**: per-residue contact counts within
    $r\le6\,\AA$ limited to hydrophobic pairs; inverse as exposure
    proxy.

5.  **KD-weighted SASA**: Kyte--Doolittle hydropathy as weights on
    per-atom SASA.

6.  **Surface charge map covariates**: net charge, pI, local
    positive/negative patch adjacency to hydrophobes (captures
    electrostatic shielding in salt).

7.  **Paratope localization indices**: fraction of hydrophobic SASA and
    aromatic SASA within CDR-H3 apex vs framework.

"'

(\[PMC\]\[2\])

"'latex

# Why these proxies should track HIC variance {#why-these-proxies-should-track-hic-variance .unnumbered}

**Hydrophobic SASA & patch size**: retention tied to hydrophobic contact
area and patch topology.

**Aromatics**: exposed F/W/Y strengthen hydrophobic and $\pi$
interactions, increasing retention.

**Charge context**: high salt screens electrostatics; residual charge
topology still modulates effective hydrophobe exposure and selectivity.

**CDR-H3 focus**: dominant contributor to extreme local hydrophobicity;
patch edits shift HIC RT. "'

(\[ScienceDirect\]\[3\])

"'latex

# Evidence that "lite" structure suffices for RT signal {#evidence-that-lite-structure-suffices-for-rt-signal .unnumbered}

**Sequence-only baselines** predict delayed HIC RT but improve with
structure-informed surface terms.

**QSAR tiers** show added value from homology/MD descriptors over
sequence alone, yet coarse structural descriptors already capture major
variance.

**Molecular-surface descriptors** (hydrophobic patches, charge patches)
correlate with multiple developability endpoints including HIC-like
risks.

**Static-structure caveat**: ensembles matter; most hydrophobic regions
dominate RT but dynamics can shift exposure. "'

(\[OUP Academic\]\[4\])

"'latex

# Operationalization on IgFold outputs {#operationalization-on-igfold-outputs .unnumbered}

1.  Build Fv with IgFold; resolve side chains; strip glycans.

2.  Compute per-atom SASA (probe 1.4 Å); tag hydrophobes and aromatics.

3.  Construct surface graph; report largest hydrophobic patch area,
    count, and density.

4.  Tally aromatic SASA and centroid clustering in CDR-H3.

5.  Compute KD-weighted SASA and hydrophobic contact density
    ($\le6\,\AA$).

6.  Map surface charge (pH 7.0--7.4); compute hydrophobe--charge
    adjacency features.

7.  Fit regularized model to HIC RT: start with elastic net on features;
    include interaction terms (aromatic_SASA $\times$ salt_type,
    charge_patch_adjacency).

"'

(\[Nature\]\[5\])

"'latex

# Confounders and controls {#confounders-and-controls .unnumbered}

Resin ligand chemistry and zeta-potential; salt type and gradient;
protein glycosylation; load and temperature; conformational ensembles
not captured by a single structure; sequence motifs creating
context-dependent exposure. *Controls*: stratify by resin; include
salt/resin covariates; sensitivity analyses with modest conformer
sampling for CDR-H3. "'

(\[SpringerOpen\]\[6\])

"'latex

# Expected signals (directional) {#expected-signals-directional .unnumbered}

-   $\uparrow$Largest hydrophobic patch area $\Rightarrow$ $\uparrow$RT.

-   $\uparrow$Aromatic SASA (F/W/Y), esp. in CDR-H3 $\Rightarrow$
    $\uparrow$RT.

-   $\uparrow$KD-weighted SASA $\Rightarrow$ $\uparrow$RT.

-   Hydrophobe--positive charge adjacency under high salt: small
    $\downarrow$RT or resin-dependent modulation.

"'

(\[ScienceDirect\]\[3\])

"'latex

# Validation plan {#validation-plan .unnumbered}

**Internal:** k-fold CV on GDP-like mAb sets; report $\Delta R^2$ of
structure-lite features over sequence baseline, SHAP for patch features.

**External:** test across resins and salts; check calibration drift.

**Ablations:** drop aromatics; drop charge-adjacency; swap IgFold with
homology model; compare to SAP-style patch metric.

**Success criterion:** robust gain in explained RT variance with minimal
overfit; consistent aromatic/CDR-H3 importance. "'

(\[PMC\]\[7\])

"'latex

# Limitations {#limitations .unnumbered}

Single static IgFold may miss exposure fluctuations; no explicit
protein--resin physics; glycan and post-translational effects excluded;
extreme sequence novelty can degrade structure quality. "'

(\[ScienceDirect\]\[8\])

"'latex

# Takeaway {#takeaway .unnumbered}

IgFold-derived surface/patch proxies are justified, cheap, and should
explain a significant fraction of HIC RT variance; aromatics and largest
hydrophobic patch dominate, with charge-context as a secondary
modulator. Prior art shows structure-informed features outperform
sequence-only baselines while avoiding MD cost; apply with resin/salt
controls and ensemble-aware caveats. "'

(\[OUP Academic\]\[4\])

\[1\]:
https://pmc.ncbi.nlm.nih.gov/articles/PMC3851231/?utm_source=chatgpt.com
\"Purification of monoclonal antibodies by hydrophobic \...\" \[2\]:
https://pmc.ncbi.nlm.nih.gov/articles/PMC11168226/?utm_source=chatgpt.com
\"Molecular surface descriptors to predict antibody \...\" \[3\]:
https://www.sciencedirect.com/science/article/abs/pii/S0021967304004121?utm_source=chatgpt.com
\"Effect of surface hydrophobicity distribution on retention \...\"
\[4\]:
https://academic.oup.com/bioinformatics/article/33/23/3758/4083264?utm_source=chatgpt.com
\"Prediction of delayed retention of antibodies in hydrophobic \...\"
\[5\]:
https://www.nature.com/articles/s41467-023-38063-x?utm_source=chatgpt.com
\"Fast, accurate antibody structure prediction from deep \...\" \[6\]:
https://bioresourcesbioprocessing.springeropen.com/articles/10.1186/s40643-024-00738-8?utm_source=chatgpt.com
\"Modeling the behavior of monoclonal antibodies on \...\" \[7\]:
https://pmc.ncbi.nlm.nih.gov/articles/PMC6410772/?utm_source=chatgpt.com
\"Five computational developability guidelines for \...\" \[8\]:
https://www.sciencedirect.com/science/article/pii/S2667119023000083?utm_source=chatgpt.com
\"Structural pre-training improves physical accuracy of \...\"
