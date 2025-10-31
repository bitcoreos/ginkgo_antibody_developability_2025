Glossary of Terms
AC-SINS (Affinity-Capture Self-Interaction Nanoparticle Spectroscopy) – A biophysical assay for antibody self-association. Antibodies are immobilized on gold nanoparticles and the plasmon wavelength shift is measured under varying salt. High self-association (often correlating with hydrophobicity) causes large plasmon shifts[1].
Antibody (Ab) – Y-shaped immunoglobulin proteins produced by B-cells, composed of two heavy and two light chains. Antibodies bind specific antigens and have variable and constant regions. They are large polypeptides with four domains per Fab arm[2]. In developability context, therapeutic IgG antibodies (especially IgG1) are evaluated for manufacturability.
Antigen – A molecule (often protein) that binds to an antibody’s variable region (paratope) to elicit an immune response[2]. (Included as background; not a direct competition element.)
AUC (Area Under the ROC Curve) – A performance metric for binary classification. It is the area under the Receiver Operating Characteristic curve, representing the probability that a classifier ranks a random positive instance higher than a random negative one. (The rules mention AUC for any binary tasks.) Citation not found in provided sources; see general ML references.
Curriculum Learning – A training strategy where a model is fed training data in order of increasing difficulty or complexity[3]. Models start on easier examples and progressively tackle harder ones, improving convergence or generalization[3].
Denaturation Midpoint (Tm) – The temperature at which 50% of protein molecules (e.g. antibody domains) are unfolded, reflecting thermostability[4]. Protein melting (Tm) is where folded and unfolded states are equally populated[4]. In antibodies, “Tm” often refers to melting of the Fab or CH2 domain.
Developability – The likelihood that a therapeutic antibody can be successfully developed. It combines intrinsic biophysical parameters (Aggregation, Solubility, Stability, Hydrophobicity, etc.) that affect manufacturability and performance[5]. Poor developability (e.g. high polyreactivity or low stability) causes clinical failures[5].
Ensemble Learning – A machine-learning technique combining multiple base models (“learners”) to improve overall predictive accuracy[6]. By aggregating diverse models (e.g. via voting or averaging), ensemble models reduce variance and bias, often outperforming single models[6].
Entropy (Information Entropy) – In information theory, a measure of uncertainty or unpredictability of a probability distribution[7]. A uniform distribution has maximum entropy (most uncertain); a peaked one has lower entropy[8][7]. In ML, output entropy can gauge model confidence. “Entropy gating” refers to filtering predictions by their entropy[7].
GDPa1 Dataset – The public training dataset (“Ginkgo Datapoints Ab developability 1”) of 246 antibodies with 10 biophysical assays (PROPHET-Ab platform). Assays include Hydrophobicity (HIC), Polyreactivity (bead-based PSP against CHO/OVA), Self-Association (AC-SINS), Thermostability (nanoDSF), Titer (expression yield) among others[9][10]. The competition focuses on five assays: HIC, PSP (polyreactivity), AC-SINS, Tm, and Titer.
GDPa1_v1.2 – A specific processed version of the GDPa1 data, with PR score corrected and AC-SINS (pH7.4) data added[11]. Contains folds including an “IgG isotype stratified” split[12].
HAC (Heparin Affinity Chromatography) – An assay where antibodies are tested for binding to immobilized heparin, indicating heparin-binding affinity (colloidal stability factor). Not a target prediction in this contest, but part of broader developability profiling[9].
HIC (Hydrophobic Interaction Chromatography) – A chromatographic assay for antibody hydrophobicity. Antibodies are run on a high-salt column to enhance hydrophobic binding; longer retention indicates higher hydrophobicity[13]. Used here as the “Hydrophobicity” score to predict.
IgG Isotype (Subclass) – Variants of IgG antibodies (e.g. IgG1, IgG2, IgG4) with different Fc-region sequences. IgG1 was used for dataset diversity; developability properties (e.g. Tm) vary by isotype. The training data is split to preserve isotype ratios[14].
JSON-LD – A Linked Data serialization format using JSON. It enables expressing data with semantic context. “JSON-LD is a method of encoding Linked Data using JSON”[15]. We use JSON-LD for schema definitions (with “@context” and “@type”).
Markov Chain – A stochastic process where the next state depends only on the current state (“memoryless” property)[16]. In sequences, a Markov model defines probabilities of each amino acid given the previous one.
Hidden Markov Model (HMM) – A Markov model with unobserved (hidden) states: observations (e.g. amino acids) are emitted by a latent Markov chain[17]. HMMs (especially profile-HMMs) are classic models for biological sequence families.
Partition/Fold (Cross-Validation) – Divisions of data for model validation. The GDPa1_v1.2 splits into 5 folds by hierarchical sequence clustering (“hierarchical_cluster_IgG_isotype_stratified_fold”) to ensure low sequence identity and balanced IgG subclasses in each fold[14]. This prevents leakage.
Polyreactivity (Non-Specificity) – The tendency of an antibody to bind multiple unrelated antigens (nonspecific binding). IgM antibodies often show polyreactivity, meaning they bind many distinct antigens[2]. Measured here by a bead-based assay against CHO membrane proteins and ovalbumin (originally “PSP” assay[18]), and denoted “PR_CHO” (was PSP_CHO) in data.
PSP Assay (Polyspecificity Particle Assay) – A high-throughput polyreactivity assay. Antibodies are immobilized on beads and incubated with a panel of polyspecificity reagents; nonspecific binding is read via fluorescence[18]. In GDPa1, reagents include CHO cell lysate (membrane proteins) and ovalbumin[18]. The competition label “PR_CHO” corresponds to this polyreactivity measure[19][9].
ROC (Receiver Operating Characteristic) and ROC AUC – A metric/curve for binary classifiers. ROC plots true positive vs false positive rate; AUC is its area. Mentioned as evaluation for any binary tasks (though primary tasks are regression). Definition omitted here (well-known).
Schema (Ontological) – Formal structure defining classes and properties. JSON-LD uses “@context” and “@type” to give semantic identity[15][20]. In this competition, we define schemas for Antibody, DevelopabilityAssay, Prediction, etc.
Self-Association – A property measuring antibody propensity to aggregate with itself. Assayed by AC-SINS (see above[1]) or by other methods. High self-association predicts poor developability (high viscosity or aggregation).
Spearman Rank Correlation (ρ) – A nonparametric metric of monotonic relationship between two ranked variables. ρ ranges from –1 to 1, measuring how well a monotonic function describes the relationship[21]. It is the primary leaderboard metric for regression targets.
Top-Decile Recall – The fraction of true positives that appear in the top 10% of model-ranked predictions. Used to emphasize high-ranking positives (e.g. worst-developability antibodies). (No direct citation found, but common in ranking metrics.)
Titer – The concentration or yield of antibody produced in expression (e.g. mg/L). Measured here by a ValitaQuant assay and predicted as “Titer”[9]. High titer indicates easier manufacturing.
Tm1 / Tm2 – Subdomains’ melting temperatures. Tm1 refers to the lower-temperature CH2 domain melt, and Tm2 to the higher-temperature Fab domain melt. In GDPa1, Tm2 (Fab) was ultimately used for prediction as “Thermostability”[9][19]. Tm values are from nanoDSF measurements.
Validation Set – The private hold-out set (80 antibodies) provided without assay values[10]. Teams submit predictions on these sequences; final scoring occurs against their withheld true values.
Protocols
Each step below outlines part of the competition workflow or related procedures, with references and known variations:
Data Acquisition & Processing – Obtain the GDPa1 dataset (246 antibody sequences with 10 developability assays[10]). Use only the first production batch for each antibody and compute median of any replicates to consolidate data[22]. Handle missing values as described (GDPa1_v1.2 cleaned PR score[11]).
Train/Test Splitting – Follow the provided fold splits. Use the hierarchical isotype-stratified clusters (5 folds) to train and internally validate models[14]. This ensures no heavy chain identity >~70% between folds and equal IgG1/4 ratios in each fold[14]. Optionally, one can also explore random or simple stratification folds, but balanced-cluster fold is recommended.
Feature Engineering – Extract features from antibody sequence (and structure, if used). Common protocol steps include:
Sequence Descriptors: Compute amino-acid composition, charge, hydrophobic moment, isoelectric point, length of CDRs, etc. These physicochemical features capture solubility and stability cues[5].
Embedding Representations: Feed sequences through pretrained protein language models (e.g. ESM, ProtTrans) to obtain high-dimensional embeddings that encode sequence semantics[23]. Use mean or CLS token vectors as features. (NLP techniques with transformers trained on large protein corpora have advanced antibody ML[23].)
Evolutionary Features: If multiple sequence alignments or homologs are available, compute conservation scores or covariance (“coupling”) features between residue pairs[24]. Coupling analysis (a form of co-evolutionary modeling) identifies residue-residue interactions important for structure/function[24].
Structural Features: Predict 3D structure (e.g. with ABodyBuilder, AlphaFold) and derive features like solvent-accessible surface area, predicted stability scores, or inter-residue contacts. As in Bashour et al.[25], 46 structure-derived developability parameters can be calculated.
Isotype/Metadata: Encode IgG subclass and whether sequences are heavy or light chain, etc. IgG1 vs IgG4 can affect thermostability and polyreactivity. Use one-hot or label encodings.
Model Training – For each target property, train regression (or classification, if thresholding) models. Protocol variations:
Baseline Regressors: Linear models (Ridge, Lasso), decision trees, Gaussian processes for a start.
Neural Networks and LMs: Use feedforward neural nets on engineered features, or fine-tune transformer language models on the sequences with regression heads (i.e. using LMs as encoders[23]).
Markov/HMM Models: As a specialized approach, train hidden Markov models (HMMs) on the sequence data to capture sequential dependency. Profiles of antibody families or mixture of HMMs could be used to score sequences. (HMMs are generative sequence models[17].)
Coupling-based Models: Use evolutionary coupling matrices as input to ML or statistical models. For example, project the direct coupling (DCA) scores or graph metrics into features.
Curriculum Learning: Schedule training samples from “easy” (e.g. high-confidence or consensus sequences) to “hard” (diverse or borderline cases). This can involve sorting training examples by antibody sub-family or predicted confidence and training progressively[3].
Ensembling: Combine multiple models (heterogeneous types) to improve stability[6]. For example, average predictions from a neural net, a gradient boosting machine, and an ensemble of LSTM models. Voting or stacking (meta-model) are common ensemble techniques.
Training Validation – Within the training set, perform cross-validation using the provided folds. Evaluate Spearman correlation on left-out fold predictions to tune hyperparameters. Ensure no data leakage (use only training folds for feature scaling, etc.). As a rule, do not peek at the private test set.
Prediction Submission – Generate predictions for all 80 held-out sequences. Assemble a submission file (usually CSV) with one row per antibody and columns for each target property (“Hydrophobicity”, “Polyreactivity”, “Self-association”, “Thermostability”, “Titer”). Follow the specified format (typically ID columns and numeric predictions). Zip as required and submit via the leaderboard platform.
Evaluation Protocol – The private test labels are withheld. After submission close, organizer evaluates each property. For regression tasks, scores are Spearman correlation on the hidden assays. For any classification subtasks, ROC AUC will be computed. For ranking-focused prize categories, top-decile recall is also reported. The final standings are based on these metrics on the private set. Ties (if any) are resolved by secondary criteria (such as run time or code footprint, per standard ML competition protocols).
Concept Graph
The following outlines the main entities and their relationships in the competition pipeline (mind-map style):
Antibody (entity)
Attributes: heavy_chain_seq, light_chain_seq, isotype, ID, metadata (e.g. source).
assays: links to DevelopabilityAssay results (HIC value, PR score, AC-SINS value, Tm, Titer) from GDPa1.
features: derived via Protocol steps – sequence features, embeddings, predicted structure, etc.
fold: assigned to a cross-validation fold ID (1–5) based on clustering.
DevelopabilityAssay (entity)
Classes: Hydrophobicity (HIC), Polyreactivity (PR_CHO), Self-Association (AC-SINS), Thermostability (Tm), Titer.
Connects Antibody → assay value (training set has values, test set withheld).
Model (agent)
Input: Antibody features.
Strategies: LMs, Markov/HMM, coupling, neural nets, ensemble, etc. (detailed in Section 7).
Output: Prediction entity with five values (predicted assay scores).
Prediction (artifact)
Schema: links antibody ID to predicted values for each property. (Output file format.)
Feeds into Evaluation (next step).
Evaluation (process)
Takes Prediction and hidden true assay values. Computes metrics: Spearman, top-decile recall, AUC[21].
Updates Leaderboard ranks.
Leaderboard (concept)
Tracks participants' performance per property. Ranks by metrics on private set.
Protocols/Steps (module)
Data processing, splitting, feature engineering, modeling, submission. Each protocol step connects entities (e.g. Antibody → features, Model → Prediction).
Relationships flow:
Antibody → (via Protocols) → Features → (via Modeling Techniques) → Predictions → (via Evaluation Metrics) → Leaderboard.
Schema Blocks
Below are simplified JSON-LD schemas for key data artifacts (context URIs are illustrative):
{
  "@context": {
    "antibody": "http://schema.org/antibody",
    "id": "@id",
    "sequence": "http://schema.org/hasPart",
    "heavy_chain": "http://schema.org/heavyChainSequence",
    "light_chain": "http://schema.org/lightChainSequence",
    "isotype": "http://schema.org/additionalProperty",
    "titer": "http://schema.org/titerValue",
    "HIC": "http://schema.org/hydrophobicityValue",
    "Polyreactivity": "http://schema.org/polyreactivityValue",
    "SelfAssociation": "http://schema.org/selfAssociationValue",
    "Thermostability": "http://schema.org/thermostabilityValue"
  },
  "@type": "antibody",
  "id": "antibody:GB_AB_001",
  "heavy_chain": "EVQLVESGGGL...",
  "light_chain": "DIQMTQSPSSLSASVGDR...",
  "isotype": "IgG1",
  "titer": 42.5,
  "HIC": 0.78,
  "Polyreactivity": 0.12,
  "SelfAssociation": 1.234,
  "Thermostability": 72.3
}
{
  "@context": {
    "prediction": "http://schema.org/prediction",
    "antibody_id": "http://schema.org/identifier",
    "pred_Titer": "http://schema.org/predictedValue",
    "pred_HIC": "http://schema.org/predictedValue",
    "pred_Polyreactivity": "http://schema.org/predictedValue",
    "pred_SelfAssociation": "http://schema.org/predictedValue",
    "pred_Thermostability": "http://schema.org/predictedValue"
  },
  "@type": "prediction",
  "antibody_id": "GB_AB_001",
  "pred_Titer": 40.1,
  "pred_HIC": 0.75,
  "pred_Polyreactivity": 0.15,
  "pred_SelfAssociation": 1.20,
  "pred_Thermostability": 70.5
}
{
  "@context": {
    "schema": "http://schema.org/",
    "ear": "http://schema.org/EvaluationAndReporting"
  },
  "@type": "schema:Evaluation",
  "schema:input": { "@id": "prediction:1234" },
  "schema:output": { 
    "spearman_Titer": 0.65,
    "spearman_HIC": 0.52,
    "spearman_Polyreactivity": 0.47,
    "spearman_SelfAssociation": 0.60,
    "spearman_Thermostability": 0.70
  }
}
Notes on schema: JSON-LD uses @context to map terms to URIs and @type for classes[15][20]. The above examples show how an Antibody object might be encoded with assay values (for training data) and how a Prediction object carries model outputs. Each class and property can be given clear IRIs for agent-based querying (e.g., using a knowledge graph engine).
Mappings
This section clarifies semantic equivalences between various names, data fields, and concepts:
Assay Name Mapping:
– “HIC” ↔ Hydrophobicity (Hydrophobic Interaction Chromatography)[13].
– “AC-SINS_pH7.4” ↔ Self-Association (Affinity Capture Self-Interaction Nanoparticle Spectroscopy at pH 7.4)[1].
– “PSP_CHO” (Polyspecificity Reagent, CHO SMP) and “PSP_OVA” ↔ Polyreactivity measures; the competition uses “PR_CHO” to denote CHO-based polyreactivity[9][19]. In code, PSP_CHO was renamed to PR_CHO (Polyreactivity) for clarity[19].
– “Tm1” vs “Tm2”: In rules “Tm1” was listed, but code expects “Tm2”. In GDPa1 context, Tm2 = Fab melting temperature[19][25]. We map Thermostability to “Tm2”.
Data vs. Ontology:
– The CSV column hierarchical_cluster_IgG_isotype_stratified_fold ↔ A splitting protocol ensuring equal IgG subclass distribution in each fold[14].
– Model features like embedding_esm1b ↔ “protein language model embedding” concept; dca_score ↔ “evolutionary coupling” feature. These map numeric columns to conceptual entities (e.g. embeddings = LMs, DCA = coupling).
Leaderboards and Metrics: The leaderboard code names metrics (spearman, auc, recall). E.g., score_titer (code) ↔ Spearman on Titer predictions. “Top-decile recall” maps to fraction of positives in top-10% predictions (definition above).
Ontology Alignments: The JSON-LD type “Antibody” maps to the concept of antibodies in ontologies; “DevelopabilityAssay” concepts align with assay names. Mapping ensures agents querying one term can retrieve related data.
Terminology Synonyms: As noted, Polyreactivity and Polyspecificity are often used interchangeably. The data rename from PSP to PR indicates these synonyms. Also, IgG1 and IgG4 are subclasses of IgG (fully synonymous in context of isotype field usage).
Constraints and Validation Rules
Human-readable constraints:
- Antibody sequences must contain only standard amino acid letters (A, C, …, Y).
- Prediction values must be numeric; expected ranges can be inferred (e.g. HIC retentions >0, PR scores ≥0).
- Submission format: One entry per antibody; must include all five predicted properties. File naming and CSV schema must follow competition specs (not detailed here).
- Data usage: Models may use only the public training data (246 seqs + assays). Private (80 seqs) and withheld assay values must not be used for training or published[26][10].
- Batch effects: Participants must ignore batch/replicate artifacts by using medians as done in data processing[22].
- Modeling rules: Any external data (e.g. protein structures) must be public as of competition start.
Logic-enforceable rules: (in pseudo-logic or Schema constraints)
- For each Antibody object a:
- a.heavy_chain_seq ≠ null, length > 0.
- a.light_chain_seq ≠ null.
- a.isotype ∈ {“IgG1”, “IgG2”, “IgG4”}.
- If using LMs: features.embedding ∈ ℝ^N, where N matches LM dimension.
- For each Prediction object p:
- p.antibody_id matches an ID in the provided test set.
- pred_X values are floats; they could optionally be constrained to [min,max] from training.
- Metric calculation constraints: For each target X, define score_X = Spearman(predictions_X, true_X). Recall can be computed as Recall@10% = (# true positives in top 10% preds) / (# total positives).
- Cross-field consistency: If an antibody is missing (no sequence), it must not appear in predictions. If a prediction is given, its antibody ID must exist in the test set.
These rules can be encoded in a schema or validation script. For example, in SHACL or JSON Schema: requiring numeric types for prediction fields, enumerated values for isotype, matching IDs between schema classes, etc.
Modeling Techniques
Protein Language Models (LMs): Pretrained transformer models (e.g. ESM, ProtBert) treat amino acid sequences like sentences[23]. They learn deep embeddings by self-supervised training on millions of protein sequences. Embeddings capture semantic and biophysical patterns (hydrophobic clusters, structural motifs, etc.)[23]. In practice, one feeds each antibody sequence to the LM and extracts a fixed-length vector (often mean of token embeddings or the special [CLS] token). This vector is used as input to downstream regressors. LMs have advanced state-of-art in many protein prediction tasks due to their unsupervised pretraining[23].
Markov Models / HMMs: Model antibody sequences as generated by a probabilistic chain. A first-order Markov model assumes each amino acid depends only on the previous one[16]. A Hidden Markov Model (HMM) introduces hidden states representing, e.g., structural motifs; emissions are the observed residues[17]. In developability modeling, one could train an HMM on antibody variable-region sequences and use its log-likelihood as a feature (e.g. a poorly fitting sequence might predict instability). Profile HMMs, common in bioinformatics, can capture conserved motifs in antibody families, potentially correlating with developability phenotypes.
Evolutionary Coupling: This technique (Direct Coupling Analysis) computes statistical couplings between pairs of residues in a multiple sequence alignment, identifying co-evolving residues[24]. Highly coupled residues often contact in 3D structure. As a modeling feature, one can derive scalar summaries of the coupling matrix (e.g. average coupling, network connectivity) or even use coupling scores as inputs to neural nets. Coupling features inject knowledge of residue interactions and natural constraints, potentially highlighting instability mutations.
Entropy-based Gating: The entropy of a model’s output distribution (across ensemble or classes) can signal confidence[7]. A gating rule might be: if entropy > threshold, downweight or exclude that prediction. For regression ensembles, an analogous “entropy” is predictive variance. Gating by uncertainty can improve reliability by focusing on high-confidence predictions. (While not standard in antibody tasks, the concept is used in adaptive systems[7].)
Curriculum Learning: As noted above[3], the idea is to feed training samples from easiest to hardest. For example, one might first train on antibodies with consensus germline sequences (less diverse), then gradually include synthetic or highly mutated sequences. Easy/hard can be defined by sequence divergence or known assay extremeness. Empirical studies have shown curricula can speed up training or find better optima compared to random sampling[3].
Ensembling: Combining multiple models is a robust way to boost performance[6]. In leaderboard competitions, top teams often train diverse models (different architectures, features, random seeds) and average or vote their predictions. The advantage is reduction of model-specific errors and variance. Common ensembles include bagging (averaging many trees), boosting (e.g. XGBoost), stacking (training a meta-learner on base model outputs), or simple arithmetic ensembles of the best-ranked models.
Each modeling choice above can be backed by literature. For instance, the utility of LMs is well-supported by recent protein ML surveys[23], and ensemble methods are a standard recommendation to improve leaderboard rankings[6]. The rationale behind each choice is either incorporation of biological insight (coupling) or ML best practices (ensembling, curriculum).
Feature Taxonomy
We categorize possible input features into broad classes:
Sequence-derived Features: Based on the primary amino-acid sequence. Includes amino acid composition (frequency of each residue), physicochemical indices (e.g. hydrophobicity scales, charge, aliphatic index), isoelectric point (pI), length of Complementarity-Determining Regions (CDRs), sequence motifs (e.g. glycosylation sites)[5]. These features capture basic structural and biochemical properties that influence developability (e.g. many hydrophobic residues → higher aggregation).
Physicochemical & Statistical Descriptors: Global numeric summaries like grand average of hydropathy (GRAVY), net charge, predicted solubility scores, aggregation-prone region scores, and disorder propensity. Many published tools (e.g. ProtParam) compute such indices[5]. They relate to expressibility and stability: e.g. high GRAVY often correlates with low solubility.
Structural Features: Derived from predicted or known 3D structure (from the Fv region). Examples: solvent accessible surface area (SASA) of paratope, buried surface area, stability free-energy estimates (ΔΔG), structural symmetry, loops conformation. Bashour et al. computed 46 structure-based developability parameters covering shape, charge distribution, cavity volume, etc.[25]. In practice, tools like Rosetta or FoldX could yield energy scores; others compute 3D moment of inertia or curvature. Structure features typically help predict thermal stability or aggregation propensity.
Embedding Features (LMs): High-dimensional vectors from pretrained models[23]. These embeddings implicitly encode many properties (structure, function motifs) without manual design. For example, an antibody’s embedding might place similar sequences (even with differing lengths) near each other in vector space, aiding regression. We treat embedding dimensions as features (dense, e.g. 768-D).
Evolutionary Features: Features based on multiple sequence alignments (MSA) of homologous antibodies. Conservation scores (entropy per position), position-specific scoring matrix values, and coupling scores (pairwise correlations) all fall here[24]. For a given antibody sequence, one can summarize conservation (e.g. average log-odds score) or count coevolving pairs. Coupling-based features aim to capture internal constraints of the sequence beyond linear composition.
Experimental Metadata: The assays measured in GDPa1 (HIC, AC-SINS, etc) are output labels, but some pipeline may use intermediate assay data (e.g. scaffold binding tests) as features in multi-task models. We note them here for completeness, though models generally predict them rather than use them as inputs.
Hierarchical/Cluster Features: The fold ID (cluster membership) or similarity metrics could be included, though careful to avoid leakage. For instance, one-hot fold ID or distance from training cluster center might encode “novelty.”
Ensemble-derived Meta-Features: Predictions of simpler baseline models (like single-task regressors) can be used as features in stacking. For example, one feature could be “prediction of a small decision tree model,” which informs a larger ensemble.
Chain-level vs. Fv-level: Some features may be computed separately for heavy and light chain (e.g. heavy-chain CDR3 length) and then concatenated. Others treat the whole Fv region.
Relation to Modeling & Performance: Good features reduce model learning burden. Sequence and physicochemical features often allow simpler models (linear/regression) to capture trends, but may miss complex patterns that LMs or coupling can catch. Structure features are powerful for thermostability and self-association but depend on accurate modeling. Embeddings can capture higher-order patterns learned from data[23], often boosting performance when training data is limited. In practice, top solutions likely use a combination (ensemble) of embedding-based models and carefully engineered descriptors, aligning with the feature diversity seen in large antibody studies[5][25].
Competency Questions
Below are example queries that the semantic mesh should support (the answers can be found by traversing the concepts and mappings):
Glossary & Concept Queries:
What does PR_CHO mean, and how is it related to polyreactivity?
Define AC-SINS and explain which developability property it measures.
What is the difference between a Markov model and a Hidden Markov Model?
What is JSON-LD used for in this context?
Explain top-decile recall and how it’s computed.
Protocol/Data Queries:
How were the cross-validation folds constructed, and why is isotype stratification important?
What processing was done on GDPa1 assay replicates?
Which columns of the data correspond to hydrophobicity, polyreactivity, and other assays?
Feature/Modeling Queries:
What kinds of features can be derived from an antibody sequence for developability prediction?
How do protein language model embeddings relate to antibody developability tasks?
Which modeling strategies gave the best leaderboard performance (e.g. ensemble of LMs)?
Why might one use curriculum learning in training these models?
Mappings/Schema Queries:
In the JSON-LD schema, which property corresponds to the antibody’s heavy-chain sequence?
How does “Polyreactivity” map to the leaderboard code name and assay?
What is the JSON-LD @type for a prediction?
Constraints/Validation Queries:
What rules must a valid prediction submission obey?
Are participants allowed to use the private test assay values during development? (Answer: No[26].)
Misc Queries:
List all developability assays measured in GDPa1[9].
Which assays did the competition choose to predict? (Answer: HIC, Polyreactivity, AC-SINS, Tm, Titer[9].)
What is the meaning of “Ontology” and how does it apply here? (OWL definition[20].)
These competency questions guide the kinds of information agents should be able to retrieve from the semantic mesh.
References Appendix
BibTeX
@misc{Wiki_JSON-LD,
  author = {Wikimedia Foundation},
  title = {JSON-LD},
  year = {2024},
  howpublished = {Wikipedia (\url{https://en.wikipedia.org/wiki/JSON-LD})}
}
@misc{Wiki_Antibody,
  author = {Wikimedia Foundation},
  title = {Antibody},
  year = {2024},
  howpublished = {Wikipedia (\url{https://en.wikipedia.org/wiki/Antibody})}
}
@misc{Wiki_Denaturation,
  author = {Wikimedia Foundation},
  title = {Denaturation (protein)},
  year = {2024},
  howpublished = {Wikipedia (\url{https://en.wikipedia.org/wiki/Denaturation_(protein)})}
}
@misc{Wiki_Spearman,
  author = {Wikimedia Foundation},
  title = {Spearman's rank correlation coefficient},
  year = {2023},
  howpublished = {Wikipedia (\url{https://en.wikipedia.org/wiki/Spearman's_rank_correlation_coefficient})}
}
@article{Estep2015,
  author = {Patricia Estep and Isabelle Caffry and Yao Yu and Tingwan Sun and Yuan Cao and Heather Lynaugh and Tushar Jain and Maximiliano V{\'a}squez and Peter M Tessier and Yingda Xu},
  title = {An alternative assay to hydrophobic interaction chromatography for high-throughput characterization of monoclonal antibodies},
  journal = {mAbs},
  year = {2015},
  volume = {7},
  number = {3},
  pages = {553--561},
  doi = {10.1080/19420862.2015.1016694}
}
@article{Makowski2021,
  author = {Emily K Makowski and Lina Wu and Alec A Desai and Peter M Tessier},
  title = {Highly sensitive detection of antibody nonspecific interactions using flow cytometry},
  journal = {mAbs},
  year = {2021},
  volume = {13},
  number = {1},
  pages = {1951426},
  doi = {10.1080/19420862.2021.1951426}
}
@article{Bashour2024,
  author = {Habib Bashour and Eva Smorodina and Matteo Pariset and Jahn Zhong and Romario Barren and Gerald Goh and Elena Romani and Paul Pugno and Lawrence Shapiro and Andrew Ward and Victor Gadgil and Angela Belgrave and Sierra Heald and Zsofia Dudas and James Moses and Jeffrey Gray},
  title = {Biophysical cartography of the native and human-engineered antibody landscapes quantifies the plasticity of antibody developability},
  journal = {Communications Biology},
  year = {2024},
  volume = {7},
  pages = {922},
  doi = {10.1038/s42003-024-06561-3}
}
@misc{GinkgoGDPa1,
  author = {Ginkgo Bioworks},
  title = {GDPa1 Antibody Developability Dataset (public training data)},
  year = {2025},
  howpublished = {Hugging Face (\url{https://huggingface.co/datasets/ginkgo-datapoints/GDPa1})}
}
@misc{GinkgoRules2025,
  author = {Ginkgo Bioworks},
  title = {Official Rules, Antibody Developability Prediction Competition 2025},
  year = {2025},
  howpublished = {Competition official rules (supabase PDF)}
}
@misc{Wiki_CurriculumLearning,
  author = {Wikimedia Foundation},
  title = {Curriculum learning},
  year = {2023},
  howpublished = {Wikipedia (\url{https://en.wikipedia.org/wiki/Curriculum_learning})}
}
@misc{Jiang2024SEER,
  author = {Jiang, Jiyuan and Others},
  title = {SEER-MoE: Sparse Expert Efficiency through Regularization},
  year = {2024},
  howpublished = {arXiv:2404.05089 [cs.LG]}
}
@misc{IBM2022Ensemble,
  author = {Murel, Jacob and Kavlakoglu, Eda},
  title = {What is Ensemble Learning?},
  year = {2022},
  howpublished = {IBM (Think) web article (\url{https://www.ibm.com/think/topics/ensemble-learning})}
}
@article{Wang2025,
  author = {Wang, Lei and Li, Xudong and Zhang, Han and Wang, Jinyi and Jiang, Dingkang and Xue, Zhidong and Wang, Yan},
  title = {A Comprehensive Review of Protein Language Models},
  journal = {arXiv preprint arXiv:2502.06881},
  year = {2025}
}
@misc{Wiki_MarkovChain,
  author = {Wikimedia Foundation},
  title = {Markov chain},
  year = {2023},
  howpublished = {Wikipedia (\url{https://en.wikipedia.org/wiki/Markov_chain})}
}
@misc{Wiki_HMM,
  author = {Wikimedia Foundation},
  title = {Hidden Markov model},
  year = {2023},
  howpublished = {Wikipedia (\url{https://en.wikipedia.org/wiki/Hidden_Markov_model})}
}
@misc{Wiki_OWL,
  author = {Wikimedia Foundation},
  title = {Web Ontology Language},
  year = {2024},
  howpublished = {Wikipedia (\url{https://en.wikipedia.org/wiki/Web_Ontology_Language})}
}
APA
Wikipedia. (2024). JSON-LD. Retrieved October 2025, from https://en.wikipedia.org/wiki/JSON-LD[15]
Wikipedia. (2024). Antibody. Retrieved October 2025, from https://en.wikipedia.org/wiki/Antibody[2]
Wikipedia. (2024). Denaturation (protein). Retrieved October 2025, from https://en.wikipedia.org/wiki/Denaturation_(protein)[4]
Wikipedia. (2023). Spearman’s rank correlation coefficient. Retrieved October 2025, from https://en.wikipedia.org/wiki/Spearman's_rank_correlation_coefficient[21]
Estep, P., Caffry, I., Yu, Y., Sun, T., Cao, Y., Lynaugh, H., Jain, T., Vásquez, M., Tessier, P. M., & Xu, Y. (2015). An alternative assay to hydrophobic interaction chromatography for high-throughput characterization of monoclonal antibodies. mAbs, 7(3), 553–561[13][1].
Makowski, E. K., Wu, L., Desai, A. A., & Tessier, P. M. (2021). Highly sensitive detection of antibody nonspecific interactions using flow cytometry. mAbs, 13(1), 1951426[18].
Bashour, H., Smorodina, E., Pariset, M., Zhong, J., Barren, R., Goh, G., … Gray, J. (2024). Biophysical cartography of the native and human-engineered antibody landscapes quantifies the plasticity of antibody developability. Communications Biology, 7, 922[5][25].
Ginkgo Bioworks. (2025). GDPa1 Antibody Developability Dataset (public training data) [Dataset]. Hugging Face. Retrieved October 2025, from https://huggingface.co/datasets/ginkgo-datapoints/GDPa1[9].
Ginkgo Bioworks. (2025). Official Rules, Antibody Developability Prediction Competition 2025. (Competition rules PDF)[10][26].
Wikipedia. (2023). Curriculum learning. Retrieved October 2025, from https://en.wikipedia.org/wiki/Curriculum_learning[3].
Jiang, J., et al. (2024). SEER-MoE: Sparse Expert Efficiency through Regularization. arXiv:2404.05089[7].
Murel, J., & Kavlakoglu, E. (2022). What is Ensemble Learning? IBM Think Blog. Retrieved October 2025, from https://www.ibm.com/think/topics/ensemble-learning[6].
Wang, L., Li, X., Zhang, H., Wang, J., Jiang, D., Xue, Z., & Wang, Y. (2025). A Comprehensive Review of Protein Language Models. arXiv:2502.06881[23].
Wikipedia. (2023). Markov chain. Retrieved October 2025, from https://en.wikipedia.org/wiki/Markov_chain[16].
Wikipedia. (2023). Hidden Markov model. Retrieved October 2025, from https://en.wikipedia.org/wiki/Hidden_Markov_model[17].
Wikipedia. (2024). Web Ontology Language. Retrieved October 2025, from https://en.wikipedia.org/wiki/Web_Ontology_Language[20].
[1] [13] An alternative assay to hydrophobic interaction chromatography for high-throughput characterization of monoclonal antibodies - PubMed
https://pubmed.ncbi.nlm.nih.gov/25790175/
[2] Antibody - Wikipedia
https://en.wikipedia.org/wiki/Antibody
[3] Curriculum learning - Wikipedia
https://en.wikipedia.org/wiki/Curriculum_learning
[4] Denaturation midpoint - Wikipedia
https://en.wikipedia.org/wiki/Denaturation_midpoint
[5] [25] Biophysical cartography of the native and human-engineered antibody landscapes quantifies the plasticity of antibody developability | Communications Biology
https://www.nature.com/articles/s42003-024-06561-3?error=cookies_not_supported&code=f3be9a40-a3f3-45b4-9273-6c6963995c69
[6] What is ensemble learning? | IBM
https://www.ibm.com/think/topics/ensemble-learning
[7] SEER-MoE: Sparse Expert Efficiency through Regularization for Mixture-of-Experts
https://arxiv.org/html/2404.05089v1
[8] Entropy (information theory) - Wikipedia
https://en.wikipedia.org/wiki/Entropy_(information_theory)
[9] [11] [12] [14] [22] ginkgo-datapoints/GDPa1 · Datasets at Hugging Face
https://huggingface.co/datasets/ginkgo-datapoints/GDPa1
[10] [26] 2025 Ginkgo Antibody Developability Prediction Competition
https://euphsfcyogalqiqsawbo.supabase.co/storage/v1/object/public/gdpweb/pdfs/2025%20Ginkgo%20Antibody%20Developability%20Prediction%20Competition%202025-08-28-v2.pdf
[15] JSON-LD - Wikipedia
https://en.wikipedia.org/wiki/JSON-LD
[16] Markov chain - Wikipedia
https://en.wikipedia.org/wiki/Markov_chain
[17] Hidden Markov model - Wikipedia
https://en.wikipedia.org/wiki/Hidden_Markov_model
[18] Overview of the PolySpecificity Particle (PSP) assay for evaluating... | Download Scientific Diagram
https://www.researchgate.net/figure/Overview-of-the-PolySpecificity-Particle-PSP-assay-for-evaluating-antibody-nonspecific_fig1_353497548
[19] Removed pycache (hopefully), and fixed assay names · ginkgo-datapoints/abdev-leaderboard at 5b5ee28
https://huggingface.co/spaces/ginkgo-datapoints/abdev-leaderboard/commit/5b5ee28229a534796c9699e410892567bf4f9beb
[20] Web Ontology Language - Wikipedia
https://en.wikipedia.org/wiki/Web_Ontology_Language
[21] Spearman's rank correlation coefficient - Wikipedia
https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
[23] A Comprehensive Review of Protein Language Models
https://arxiv.org/html/2502.06881v1
[24]  Evolutionary coupling analysis guides identification of mistrafficking-sensitive variants in cardiac K+ channels: Validation with hERG - PMC 
https://pmc.ncbi.nlm.nih.gov/articles/PMC9632996/
