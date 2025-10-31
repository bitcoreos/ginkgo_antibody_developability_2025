# Entropy Study: Polyspecificity Risk Stratification Across Surprisal Tiers {#entropy-study-polyspecificity-risk-stratification-across-surprisal-tiers .unnumbered}

## Scope {#scope .unnumbered}

Goal: map sequence-information content (surprisal) to polyspecificity
risk classes, grounded in empirical determinants of antibody
polyreactivity and nonspecific binding (PSR/PSP, heparin/lysate binding,
charge/hydrophobe patches) and paratope dynamics literature.

## Definitions {#definitions .unnumbered}

**Polyreactivity / polyspecificity**: non-specific or multi-antigen
binding that increases clearance and safety risk in therapeutics
[@Cunningham2021; @Herling2023; @Chen2024].\
**PSR/PSP**: high-throughput assays quantifying nonspecific binding to
complex reagents or particles [@Xu2013; @Makowski2021].\
**Local surprisal**: $S_k(i)=-\log p(s_{i..i+k-1})$; negative
log-probability of a $k$-mer under a background model estimated over
human IG variable-region repertoires (or curated developable sets). High
$S_k$ indicates locally rare motifs [@Humphrey2020].\
**Entropy**: Shannon sequence entropy over sliding windows (proxy for
compositional variability); complements surprisal [@Anashkina2021].

## Biophysical priors linking sequence features  polyreactivity {#biophysical-priors-linking-sequence-features-polyreactivity .unnumbered}

-   Heavy chain dominance: V~H~ drives most polyreactivity; risk rises
    with positive charge and hydrophobicity (and combinations)
    [@Chen2024].

-   Charge patterning: elevated Lys/Arg, high pI, and basic patches
    promote binding to anionic species (DNA, heparin, lysate components)
    [@Lecerf2023; @Cunningham2021].

-   Hydrophobic patches: surface hydrophobe clusters control nonspecific
    binding and even phase separation [@Ausserwoger2023].

-   Dynamics: polyreactivity associates with flexible, multi-state
    paratopes; affinity maturation tends to rigidify and reduce
    polyreactivity
    [@Shehata2019; @FQ2020states; @FQ2020rigid; @Jeliazkov2018; @Guthmiller2020; @Prigent2018].

-   Sequence motifs: bioinformatic analyses implicate increased
    aromatics/hydrophobes and basic residues in polyreactive sets
    [@Boughter2020; @Lecerf2023].

## Hypothesis {#hypothesis .unnumbered}

Local sequence rarity (high $S_k$) indicates atypical physico-chemical
microcontexts (e.g., cationic/hydrophobic clusters or unusual loop
chemistries) that correlate with: (i) larger positive/negative
surface-patch imbalance, (ii) higher conformational heterogeneity, and
thus (iii) increased PSR/PSP signal and off-target binding risk.
Prediction: monotone non-decreasing relationship between top-quantile
$S_k$ burden and polyreactivity readouts, modulated by net charge and
patchiness. (Mechanistic support: charge/hydrophobe patches
[@Chen2024; @Ausserwoger2023]; flexibility--polyreactivity linkage
[@Shehata2019; @FQ2020states; @Guthmiller2020].)

## Surprisal-tiering protocol (sequence-only; chain-specific) {#surprisal-tiering-protocol-sequence-only-chain-specific .unnumbered}

Background model: IGHV/IGKV human repertoire Markov model (order 1--3)
or empirical $k$-mer table; $k\in[3,6]$ recommended [@Humphrey2020].\
**Compute:** for each position/window, $S_k(i)$; summarize per chain by:
$$\text{Burden}_{q} = \frac{1}{L}\sum_{i} \mathbb{I}\{S_k(i)\ge \text{Quantile}_q\},\quad q\in\{90,95,99\}.$$
$$\text{S-mean}=\frac{1}{L}\sum_i S_k(i),\quad \text{S-max}=\max_i S_k(i).$$
**Tiers (per chain, conservative):**\
T0 (Baseline): $\text{Burden}_{90}<0.05$ and S-mean $<\mu+\sigma$.\
T1 (Mild): $0.05\le\text{Burden}_{90}<0.10$ or S-mean
$\in[\mu+\sigma,\mu+2\sigma)$.\
T2 (Moderate): $\text{Burden}_{95}\ge0.05$ or S-max
$\ge \text{Q}_{99}$.\
T3 (High): $\text{Burden}_{99}\ge0.02$ or ( $\ge \mu+2\sigma$ and
$\text{Burden}_{95}\ge0.10$).\
(*Notes*: $\mu,\sigma,\text{Q}_p$ derived from a reference set of
low-PSR clinical-stage mAbs; thresholds are to be calibrated on labeled
PSR/PSP panels [@Makowski2021; @Herling2023].)

## Risk model (integrates priors + tiers) {#risk-model-integrates-priors-tiers .unnumbered}

$$\text{logit}(\Pr[\text{polyreactive}]) = \beta_0 + \beta_1 \text{Tier}_{VH} + \beta_2 \text{Tier}_{VK} + \beta_3 \text{NetCharge}_{VH} + \beta_4 \text{HydropPatch}_{VH} + \beta_5 \text{pI}_{VH} + \beta_6 \text{CDR\!-\!H3Len} + \beta_7 \text{FlexProxy},$$
where *HydropPatch* = largest Kyte--Doolittle patch on solvent-exposed
surface (from fast model or proxy descriptors), *FlexProxy* = entropy of
predicted paratope states or loop RMSF proxy; include interaction
(*NetCharge*$\times$Tier) given heavy-chain mediation
[@Chen2024; @Shehata2019; @FQ2020states]. Regularize with monotonic
constraints on $\beta_1,\beta_3,\beta_4$.

## Assay mapping and decision rules {#assay-mapping-and-decision-rules .unnumbered}

-   **Screen**: PSR/PSP flow assays (lysate, DNA, heparin)
    [@Xu2013; @Makowski2021]; orthogonal microfluidic NSB fingerprints
    [@Herling2023]; heparin/HIC as confirmatory
    [@Cunningham2021; @Ausserwoger2023].

-   **Cut-lines (to be fit)**: choose PSR MFI thresholds that maximize
    Youden's J vs. tiered predictions on a clinical-stage set
    [@Makowski2021; @Herling2023].

-   **Escalation**: Tier T2--T3 and/or high PSR $\Rightarrow$ charge
    trimming (reduce Lys/Arg clusters), de-aromatize exposed patches, or
    maturation-driven rigidification (sequence-level)
    [@Shehata2019; @Cunningham2021; @Lecerf2023].

## Validation plan {#validation-plan .unnumbered}

**Data**: assemble $\geq$`<!-- -->`{=html}200--400 mAbs with PSR/PSP and
heparin/HIC labels (public + internal).\
**Models**: baseline logistic vs. isotonic add-on for tiers; nested
models with and without surprisal features to test incremental AUC,
PR-AUC, calibration, and partial-AUC at low false-positive rates.\
**Ablations**: swap background models (IGH repertoire vs. therapeutic
set), vary $k$, chain-only vs. combined, remove charge/hydrophobe
covariates to isolate surprisal signal.\
**Mechanistic checks**: enrichment of high-surprisal windows in CDR-H3
and basic/hydrophobe motifs [@Boughter2020; @Lecerf2023]; correlation of
tier with FlexProxy [@FQ2020states; @Jeliazkov2018].\
**Prospective**: blinded prospective PSR on 50 designs spanning tiers
T0--T3; pre-registered thresholds.

## Interpretation {#interpretation .unnumbered}

-   Surprisal alone is not causal; it priors local rarity that
    co-travels with risky chemistries and dynamics
    [@Humphrey2020; @Chen2024; @Ausserwoger2023].

-   Expect heaviest lift from V~H~ tier, net positive charge, and
    hydrophobe-patch metrics; dynamics proxies add in flexible families
    [@Chen2024; @Shehata2019; @FQ2020states].

-   Affinity maturation or targeted edits that reduce tiered burden
    should reduce PSR [@Shehata2019; @Cunningham2021].

## Limitations {#limitations .unnumbered}

Calibration requires matched assay conditions; background models
sensitive to training corpora; dynamics proxies are approximations;
direct causality unproven---framework is risk stratification, not
mechanistic proof [@Cunningham2021; @Herling2023].

## Minimal methods (reproducible sketch) {#minimal-methods-reproducible-sketch .unnumbered}

Build $k$-mer tables from IMGT human IGHV/IGKV. Compute $S_k$ on
V-regions, summarize burdens and S-mean/S-max. Derive priors (charge,
pI, hydropatch) from sequence. Fit monotone-regularized logistic with
5-fold stratified CV. Report performance with stratified confidence
intervals. Release code and reference distributions.

## References {#references .unnumbered}

::: small
::: thebibliography
99 Makowski EK et al. Highly sensitive detection of antibody nonspecific
interactions using flow cytometry. *mAbs* (2021).
<https://pmc.ncbi.nlm.nih.gov/articles/PMC8317921/> Xu Y et al.
Addressing polyspecificity of antibodies selected from an in vitro
library. *Protein Eng Des Sel* (2013).
<https://academic.oup.com/peds/article/26/10/663/1513811> Cunningham O
et al. Polyreactivity and polyspecificity in therapeutic antibody
development. *mAbs* (2021).
<https://pmc.ncbi.nlm.nih.gov/articles/PMC8726659/> Herling TW et al.
Nonspecificity fingerprints for clinical-stage antibodies. *PNAS*
(2023). <https://www.pnas.org/doi/10.1073/pnas.2306700120> Chen HT et
al. Human antibody polyreactivity is governed primarily by the heavy
chain. *Cell Reports* (2024).
<https://www.sciencedirect.com/science/article/pii/S2211124724011525>
Lecerf M et al. Polyreactivity of antibodies from different B-cell
populations. *Front Immunol* (2023).
<https://www.frontiersin.org/articles/10.3389/fimmu.2023.1266668/full>
Ausserwöger H et al. Surface patches induce nonspecific binding and
phase separation. *PNAS* (2023).
<https://www.pnas.org/doi/10.1073/pnas.2210332120> Shehata L et al.
Affinity maturation enhances specificity but compromises conformational
stability. *Cell Reports* (2019).
<https://www.cell.com/cell-reports/pdf/S2211-1247(19)31104-0.pdf>
Fernández-Quintero ML et al. Local and global rigidification upon
antibody affinity maturation. *Front Mol Biosci* (2020).
<https://pmc.ncbi.nlm.nih.gov/articles/PMC7426445/> Fernández-Quintero
ML et al. Antibodies exhibit multiple paratope states influencing
V~H~/V~L~ dynamics. *Commun Biol* (2020).
<https://www.nature.com/articles/s42003-020-01319-z> Jeliazkov JR et al.
Repertoire analysis suggests rigidification reduces entropic losses.
*Front Immunol* (2018).
<https://www.frontiersin.org/articles/10.3389/fimmu.2018.00413/full>
Guthmiller JJ et al. Polyreactive B cells selected to provide weakly
cross-reactive immunity. *Immunity* (2020).
<https://www.cell.com/immunity/fulltext/S1074-7613(20)30446-5> Prigent J
et al. Conformational plasticity in broadly neutralizing HIV-1
antibodies. *Cell Reports* (2018).
<https://www.sciencedirect.com/science/article/pii/S2211124718306995>
Boughter CT et al. Biochemical patterns of antibody polyreactivity
revealed via CDR analysis. *eLife* (2020).
<https://elifesciences.org/articles/61393> Humphrey S et al. k-mer
surprisal to quantify local sequence complexity. *Bioinformatics*
(2020). <https://pmc.ncbi.nlm.nih.gov/articles/PMC7648452/> Anashkina AA
et al. Entropy analysis of protein sequences. *Biophys Physicobiol*
(2021). <https://pmc.ncbi.nlm.nih.gov/articles/PMC8700119/>
:::
:::
