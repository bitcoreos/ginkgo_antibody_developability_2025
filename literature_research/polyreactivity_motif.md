# Motif Atlas: Basic Residue Run-Length Patterns that Trigger QA Escalations {#motif-atlas-basic-residue-run-length-patterns-that-trigger-qa-escalations .unnumbered}

## Scope {#scope .unnumbered}

Primary sequence motifs with enriched basic residues (K, R, H) that
elevate risk of expression failure, proteolysis, non-specific
binding/polyreactivity, or unintended cellular trafficking.
Evidence-focused; minimal heuristics for automated flags.

## Notation {#notation .unnumbered}

B ≔ K,R; regex-like patterns in `monospace`. CDR ≔ IMGT CDRs for
antibodies.

## Summary Table (flag rules + rationale) {#summary-table-flag-rules-rationale .unnumbered}

+----------------+----------------+----------------+----------------+
| **Class**      | **Pattern      | **Primary Risk | **Flag         |
|                | (example       | Mechanism**    | Heuristic**    |
|                | regex)**       |                |                |
+:===============+:===============+:===============+:===============+
| Polybasic      | `K{9,}` or     | Ribosome       | Hard flag if   |
| stretch        | `R{9,}`        | stalling on    | any `K` or `R` |
| (tr            | (amino-acid    | pol            | run length     |
| anslation/RQC) | level)         | y(A)/polybasic | $\ge$`<!--     |
|                |                | tracts →       |  -->`{=html}9. |
|                | *Note:         | ribosome       | If present,    |
|                | codon-level    | collisions →   | **escalate     |
|                | poly(A)        | RQC, aborted   | codon check**: |
|                | (AAA)          | translation,   | AAA-rich vs    |
|                | $\ge$`<!-- --> | low yield      | AAG; document  |
|                | `{=html}9--12* | [@Matsuo2017;  | RQC risk.      |
|                |                |  @Tesina2019;  |                |
|                |                | @DOrazio2021]. |                |
+----------------+----------------+----------------+----------------+
| NLS-like basic | `              | Importin-      | Soft flag if   |
| cluster        | K(K|R)X(K|R)`; | $\alpha/\beta$ | motif present; |
| (monopartite)  | exemplar SV40  | recognition →  | **hard flag**  |
|                | T-Ag `PKKKRKV` | nuclear        | if             |
|                | (5 contiguous  | localization;  | $\ge$`<!-      |
|                | basic)         | in biologics,  | - -->`{=html}4 |
|                |                | potential      | contiguous B   |
|                |                | off-target     | in             |
|                |                | uptake or      | solvent-       |
|                |                | altered        | exposed/linker |
|                |                | trafficking    | regions.       |
|                |                | [@Hodel20      |                |
|                |                | 01; @NguyenBa2 |                |
|                |                | 009; @Lu2021]. |                |
+----------------+----------------+----------------+----------------+
| NLS-like basic | `K             | Same as above  | Soft flag;     |
| cluster        | RX{10-12}KRXK` | (two basic     | **hard flag**  |
| (bipartite)    |                | clusters with  | if exposed and |
|                |                | 10--12 aa      | within         |
|                |                | spacer)        | flexible       |
|                |                | [@Lu2021; @    | linkers.       |
|                |                | NguyenBa2009]. |                |
+----------------+----------------+----------------+----------------+
| Hepa           | `XBBXBX` or    | GAG binding →  | Hard flag if   |
| rin/HS-binding | `XBBBXXBX`     | non-specific   | motif occurs   |
| basic cluster  | (B=K/R)        | interactions,  | in CDRs or     |
| (Card          |                | PSR/PSP        | exposed        |
| in--Weintraub) |                | positives,     | linkers;       |
|                |                | clearance risk | require        |
|                |                | [@CardinWeintr | PSP/PSR check  |
|                |                | aub; @Fromm199 | and            |
|                |                | 7; @PLOS2019]. | HIC/AC-SINS    |
|                |                |                | context.       |
+----------------+----------------+----------------+----------------+
| Consecutive    | `R{2,}` within | Positive       | Hard flag if   |
| arginines in   | CDRs (esp. H3) | charge patches | `RRR` in any   |
| CDRs           |                | drive          | CDR or net     |
|                |                | non-specific   | positive CDR   |
|                |                | and            | patch; couple  |
|                |                | self           | to PSP/PSR and |
|                |                | -interactions; | se             |
|                |                | viscosity,     | lf-association |
|                |                | clearance      | assays.        |
|                |                | penalties      |                |
|                |                | [@WolfPerez20  |                |
|                |                | 22; @Lecerf202 |                |
|                |                | 3; @Chen2024]. |                |
+----------------+----------------+----------------+----------------+
| PC/furin       | `R-X-(K|R)-R`  | Proprotein     | Hard flag if   |
| cleavage site  | (P4--P1)       | convertase     | motif is       |
|                |                | cleavage in    | s              |
|                |                | secretory      | olvent-exposed |
|                |                | pathway →      | and in         |
|                |                | clipping of    | junctions;     |
|                |                | linkers/       | recommend PCSK |
|                |                | hinges/fusions | susceptibility |
|                |                | [@             | modeling or in |
|                |                | Garten2018; @L | vitro protease |
|                |                | ubinski2022; @ | challenge.     |
|                |                | UniProtFurin]. |                |
+----------------+----------------+----------------+----------------+
| Histidine      | `H{4,}` or     | pH-dependent   | Soft flag if   |
| clusters (pH   | dense H in     | binding and    | H-cluster in   |
| labile)        | CDRs           | assay          | CDRs;          |
|                |                | artifacts; can | **escalate**   |
|                |                | inflate        | if low-pH      |
|                |                | polyreactivity | steps exist    |
|                |                | after low-pH   | (Protein A     |
|                |                | exposure       | elution, VI).  |
|                |                | [@Klaus2021; @ |                |
|                |                | Schroter2015;  |                |
|                |                | @Arakawa2023]. |                |
+----------------+----------------+----------------+----------------+

## Detection Rules (minimal, automatable) {#detection-rules-minimal-automatable .unnumbered}

-   **Runs**: scan for `K{n,}`, `R{n,}`. Set $n{=}9$ for RQC concern;
    $n{=}4$ for trafficking/non-specificity concern.

-   **NLS**: search `K(K|R)X(K|R)`, `KRX{10-12}KRXK`; require solvent
    exposure or linker context for hard flag.

-   **GAG-binding**: sliding windows (6--8 aa) matching
    `XBBXBX`/`XBBBXXBX`.

-   **CDR-focused arginine**: locate `R{2,}` inside IMGT CDRs; combine
    with local pI and electrostatic patch metrics.

-   **Furin**: search `R-X-(K|R)-R`; map to junctions/hinge/linker;
    check accessibility.

-   **Histidine**: `H{4,}` in CDRs or clusters with nearby acidic
    residues; annotate pH-lability risk.

## Why QA Escalates {#why-qa-escalates .unnumbered}

1.  **Expression risk**: poly(A)/polybasic runs cause ribosome
    collisions and RQC; Matsuo quantified thresholds
    ($>$`<!-- -->`{=html}8--12 K codons) linking length to repression
    [@Matsuo2017]; poly(A) tracts stall eukaryotic ribosomes
    [@Tesina2019; @DOrazio2021].

2.  **Non-specificity/PSR/PSP**: positive patches and consecutive
    arginines promote nonspecific binding and self-association impacting
    viscosity and PK [@WolfPerez2022; @Lecerf2023; @Chen2024].

3.  **Trafficking**: classical NLS motifs with 4--5 contiguous basics
    (SV40 PKKKRKV) or bipartite clusters drive nuclear import via
    importin-$\alpha/\beta$ [@Hodel2001; @NguyenBa2009; @Lu2021].

4.  **Matrix binding**: Cardin--Weintraub basic patterns confer
    heparin/HS binding [@CardinWeintraub; @Fromm1997; @PLOS2019].

5.  **Proteolysis**: RX(K/R)R motifs are preferred furin/PC cleavage
    sites in the secretory route
    [@Garten2018; @Lubinski2022; @UniProtFurin].

6.  **pH-labile binding**: histidine-enriched CDRs can create
    pH-dependent binding and acid-exposure-induced polyreactivity
    [@Klaus2021; @Schroter2015; @Arakawa2023].

## Escalation Matrix (actionable) {#escalation-matrix-actionable .unnumbered}

-   **Run-length $\ge$`<!-- -->`{=html}9 K/R**: classify as *RQC-high*;
    codon audit; redesign or recode.

-   **CDR `RRR` or 4+ contiguous B anywhere exposed**:
    *Non-specificity-high*; do PSR/PSP, SIC/AC-SINS, HIC; mutate to
    neutral/acidic.

-   **NLS patterns in linkers/termini**: *Trafficking risk*; mutate to
    break motif.

-   **GAG motifs in CDRs/linkers**: *Matrix-binding risk*; test
    heparin/HS ELISA; reduce B density/spacing.

-   **Furin motif at junctions**: *Proteolysis risk*; mutate P1--P4 to
    break RX(K/R)R; protease challenge study.

-   **H clusters + low-pH processing**: *pH artifact risk*; neutral-pH
    elution/VI; histidine rational edits if needed.

## Key References {#key-references .unnumbered}

::: thebibliography
99 Matsuo Y *et al.* "Ubiquitination of stalled ribosome..." *Nat
Commun* 2017. Evidence that $>$`<!-- -->`{=html}8--12 lysine codons
repress translation $\sim$`<!-- -->`{=html}20-fold in yeast. Tesina P
*et al.* "Molecular mechanism of translational stalling by inhibitory
codon pairs and poly(A)." *EMBO J* 2019. D'Orazio KN & Green R.
"Ribosome states signal RNA quality control." *Mol Cell* 2021. Hodel MR
*et al.* "Dissection of a nuclear localization signal." *J Biol Chem*
2001. SV40 PKKKRKV as canonical monopartite NLS. Nguyen Ba AN *et al.*
"A simple HMM for NLS." *BMC Bioinformatics* 2009. cNLS consensus
K(K/R)X(K/R) and KRX$_{10–12}$KRXK. Lu J *et al.* "Types of nuclear
localization signals." *Cell Commun Signal* 2021. Cardin AD & Weintraub
HJ. "Molecular modeling of proteoglycan-protein interactions."
*Arteriosclerosis* 1989. Cardin--Weintraub XBBXBX/XBBBXXBX. Fromm JR *et
al.* "Pattern and spacing of basic amino acids in heparin binding
sites." *Arch Biochem Biophys* 1997. Pijuan-Sala B *et al.* "HS-binding
domain mapping with CW motifs." *PLoS ONE* 2019. Wolf Pérez AM *et al.*
"Assessment of therapeutic antibody developability..." 2022. Notes that
multiple consecutive arginines facilitate nonspecific interactions.
Lecerf M *et al.* "Polyreactivity of antibodies... distinct sequence
patterns." *Front Immunol* 2023. Arg enrichment in HCDR3 correlates with
polyreactivity. Chen Y *et al.* "Multi-objective engineering of
therapeutic antibodies." PhD thesis + Nat Biomed Eng 2023 derivative.
Positive CDR charge correlates with non-specificity. Garten W *et al.*
"Proprotein convertases." *Int J Mol Sci* 2018. Furin prefers RX(K/R)R.
Lubinski B *et al.* "Intrinsic furin-mediated cleavability S1/S2." *J
Virol* 2022. Motif P4--P1 R-X-(K/R)-R. UniProt Furin (P09958): consensus
RX(K/R)R and substrates. Klaus T *et al.* "pH-responsive antibodies." *J
Biomed Sci* 2021. Schröter C *et al.* "Generic approach to engineer
antibody pH-switches." *mAbs* 2015. Arakawa T *et al.* "Mechanistic
insight into poly-reactivity..." *Antibodies* 2023. Acid exposure and
histidine contexts can elevate polyreactivity.
:::
