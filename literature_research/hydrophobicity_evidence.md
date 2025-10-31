# Evidence Scan: CDR-H3 Aromatic Cluster Signatures Tied to Aggregation Alerts {#evidence-scan-cdr-h3-aromatic-cluster-signatures-tied-to-aggregation-alerts .unnumbered}

## Executive Signals {#executive-signals .unnumbered}

-   Mechanism: **surface-exposed aromatic clusters** (W/Y/F) in CDR-H3
    drive *hydrophobic patching*, $\pi$--$\pi$ stacking, and nonspecific
    contacts $\Rightarrow$ elevated self-association, viscosity, HMW
    species, and delayed HIC retention
    [@Hebditch2019; @Jain2017; @Geoghegan2016; @Park2024; @Phan2022].

-   Empirics: **HIC RT** increases with CDR aromatic content;
    **AC-SINS** shifts with self-association; **CIC** correlates with
    HIC for hydrophobic/nonspecific binders
    [@Hebditch2019; @Jain2017; @Waibl2021; @Kohli2015].

-   Sequence-level coupling: APRs frequently overlap CDRs; **H3 motifs
    enriched in W and Y** associate with nonspecificity and aggregation
    risk [@Wang2010; @Kelly2017].

## Mechanistic Rationale {#mechanistic-rationale .unnumbered}

Aromatics in H3 elevate local nonpolar surface density; planar rings
enable $\pi$--$\pi$ interactions and cation--$\pi$ contacts, stabilizing
weak, multivalent Fv--Fv networks. In high-salt HIC/AC-SINS conditions,
lyotropic screening exposes hydrophobic patches, amplifying retention or
plasmon shifts. Sequence composition $\rightarrow$ surface patches
$\rightarrow$ assay alerts
[@Hebditch2019; @Jain2017; @Phan2022; @Waibl2021].

## Empirical Links (Assay $\leftrightarrow$ H3 Aromatics) {#empirical-links-assay-leftrightarrow-h3-aromatics .unnumbered}

-   **HIC** (salt-gradient RT): positive association with [CDR aromatic
    fraction]{.underline}, non-linear with charge; models using CDR
    aromatic content predict delayed RT [@Hebditch2019; @Jain2017].

-   **AC-SINS/PS-SINS**: plasmon peak shifts up with increased
    self-association; engineered reduction of hydrophobic/aromatic
    patches reduces self-association [@Phan2022; @Geoghegan2016].

-   **CIC**: increased cross-interaction for hydrophobic/nonspecific
    paratopes; correlates with delayed HIC in discovery panels
    [@Kohli2015; @Jain2017].

-   **Viscosity/high concentration**: solvent-exposed H3 aromatics raise
    k$_\text{self}$ and viscosity; targeted mutagenesis of W/Y in CDRs
    reduces viscosity while preserving potency
    [@Park2024; @Dai2024; @Geoghegan2016].

## Sequence Signatures (H3) {#sequence-signatures-h3 .unnumbered}

-   **Aromatic load**:
    $\phi_{H3}^{\text{arom}}=\frac{\#\{W,Y,F\}}{L_{H3}}$. Risk
    heuristic: $\phi_{H3}^{\text{arom}}\ge 0.30$ with predicted solvent
    exposure for $\ge 2$ aromatics
    [@Hebditch2019; @Jain2017; @Park2024].

-   **Contiguity/cluster**: contiguous \[WYF\]{2,3}, or central H3
    triads (positions 99--101 Kabat) containing $\ge 2$ aromatics;
    enrichment of multi-Trp motifs in H3 drives nonspecificity
    [@Kelly2017].

-   **Patch geometry**: aromatic sidechains forming a convex/nonplanar
    patch at paratope; sequence features outperform static nonpolar area
    for HIC and HMWS correlation [@Hebditch2019].

-   **APR overlap**: CDR-local APRs enriched in Trp/Tyr; H3 APR exposure
    couples binding and aggregation liabilities [@Wang2010].

## Alert Mapping (Rules-of-Thumb) {#alert-mapping-rules-of-thumb .unnumbered}

::: center
  **Signature (H3)**                                                                   **Interpretation**                            **Likely Alert**
  ------------------------------------------------------------------------------------ --------------------------------------------- ------------------------------------------------------------------------------------------------------
  $\phi_{H3}^{\text{arom}}\ge 0.30$ and SAP/SASA indicates $\ge 2$ exposed aromatics   Hydrophobic/aromatic paratope patch           Delayed HIC RT; elevated CIC; AC-SINS shift [@Hebditch2019; @Jain2017]
  {2,3} contiguous or WxxW/WYxY near H3 center                                         $\pi$--$\pi$ clusters stabilize weak Fv--Fv   AC-SINS up; viscosity increase at $\ge$`<!-- -->`{=html}100 mg/mL [@Kelly2017; @Phan2022; @Park2024]
  Central H3 triad with $\ge$`<!-- -->`{=html}2 aromatics + positive flank (Arg/Lys)   Cation--$\pi$ aided patching                  CIC/HIC both high [@Kohli2015; @Jain2017]
  APR predicted within H3 containing W/Y                                               Overlap binding/aggregation hotspots          HMW species on stability stress, PEG LLPS risk [@Wang2010; @Hebditch2019]
:::

## Case Evidence {#case-evidence .unnumbered}

**Engineered paratopes**: Reducing hydrophobic/aromatic variables in
V-domains lowers AC-SINS signal and viscosity without potency loss
[@Geoghegan2016; @Dai2024]. **Library-level H3 motifs**: Trp-rich H3
centers enrich nonspecific binders; multi-Trp needed to drive phenotype
[@Kelly2017]. **Panel analytics**: CDR aromatic content explains HIC
variability; charge modulates nonlinearly [@Hebditch2019; @Jain2017].

## Screening Protocol (minimal) {#screening-protocol-minimal .unnumbered}

1.  Compute $\phi_{H3}^{\text{arom}}$, detect \[WYF\]{2,3}, WxxW/WYxY;
    predict SASA/SAP for aromatics [@Jain2017; @Park2024].

2.  Flag if: $\phi_{H3}^{\text{arom}}\ge 0.30$ *and* exposed aromatics
    $\ge 2$ *or* any contiguous \[WYF\]{2,3}.

3.  Run HIC RT, AC-SINS (salt-matched), CIC; confirm co-alerts. If
    flagged, mutate to polar aromatics (Y$\rightarrow$S/T/N/Q) or reduce
    contiguity; re-test [@Geoghegan2016; @Dai2024].

## Limits {#limits .unnumbered}

Assay and buffer context matter; HIC$\leftrightarrow$CIC correlations
vary by panel; paratope engineering must preserve affinity/epitope
geometry [@Jain2017; @Waibl2021].

# References {#references .unnumbered}

::: small
::: thebibliography
99 Hebditch M, Roche A, Curtis RA, Warwicker J. *J Pharm Sci*.
2019;108(4):1434--1441. doi:10.1016/j.xphs.2018.11.035. Jain T, et al.
*Bioinformatics*. 2017;33(23):3758--3766.
doi:10.1093/bioinformatics/btx519. Dobson CL, et al. *Sci Rep*.
2016;6:38644. doi:10.1038/srep38644. Park S, et al. *MAbs*.
2024;16(1):2346072. doi:10.1080/19420862.2024.2346072. Phan TTQ, et al.
*Biotechnol Prog*. 2022;38(5):e3305. doi:10.1002/btpr.3305. Waibl F,
et al. *Biophys J*. 2021;120(4):740--752. doi:10.1016/j.bpj.2020.12.020.
Kohli N, et al. *MAbs*. 2015;7(4):752--758.
doi:10.1080/19420862.2015.1048414. Wang X, et al. *MAbs*.
2010;2(5):452--470. doi:10.4161/mabs.2.5.12645. Kelly RL, et al. *Nat
Biomed Eng*. 2017;1:979--989. doi:10.1038/s41551-017-0161-7. Dai W,
et al. *MAbs*. 2024;16(1):2304363. doi:10.1080/19420862.2024.2304363.
:::
:::
