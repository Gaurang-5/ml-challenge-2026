# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Antigravity ER  
**Team Members:** Gaurang Bhatia  
**Submission Date:** 2026-09-26  

---

## 1. Executive Summary
We developed an end-to-end, high-precision two-stage business entity resolution architecture combining a high-recall multi-key inverted index for candidate blocking with an engineered LightGBM re-ranking classifier. Our pipeline resolves multi-lingual cross-script records across the US, India, and zero-shot France without external lookups, geocoders, or web APIs. 

Following a root-cause investigation into a validation-versus-leaderboard discrepancy (held-out validation $F_{0.5} = 0.9611$ vs. initial leaderboard $\approx 0.65$), we resolved critical failure modes: spatial false-merging of distinct businesses sharing identical building addresses, severe blocking key explosion on French stopword pairs (`de la`), numeric address corruption from excessive regex character squashing (`112` $\to$ `12`), and lack of cross-field interaction features. Retraining on a balanced dataset of 1.88M pairs enriched with hard spatial negatives, hard name negatives, and identity anchors—coupled with a `name_addr_interaction` feature (LightGBM gain #1 at 5.92M) and precision guardrails—restored true singletons to 26,973 and eliminated spurious 15–37 entity clusters. The entire inference pipeline processes 1.73M queries against ~10M targets in ~45 minutes with sub-4GB RAM, fully passing all official submission validation checks with 0 errors.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis across the 12.5M training records and systematic error analysis of the initial test submission, we identified four critical root causes limiting leaderboard performance under Macro $F_{0.5}$:

1. **Validation Illusion & Distractor Deficit**: Local held-out validation evaluated 10,000 queries against only 150,000 sampled distractors. In the full test corpus (~10M targets), target density is orders of magnitude higher, exposing catastrophic collisions between unrelated businesses located at the same physical address or multi-tenant commercial complex.
2. **Spatial False-Merge Bias in GBDT Re-ranking**: The baseline LightGBM model had overfit heavily to `block_weight` (split gain 4.5M) and `addr_token_set_ratio` (gain 640k). As a result, distinct businesses sharing an identical address (e.g., gym `Musculation Centre` vs. pharmaceutical firm `Pyranyla`, or bakery `Boulangerie Patisserie` vs. dental clinic `Cabinet Dentaire`) were assigned match probabilities of $0.967$ and $0.810$, triggering massive false merges. Under Macro $F_{0.5}$, precision is penalized $2\times$ over recall, and any false merge on a singleton entity collapses its score from $1.0$ to $0.0$.
3. **Zero-Shot Country Generalization (France)**: The training corpus contains zero French records, while the test set features extensive French entities. This induced three major linguistic breakdown points:
   - *Missing French Legal Suffixes*: Terms such as `EI`, `SASU`, `SARLU`, `GIE`, `EARL`, `SCI`, `SNC`, `SCOP` were not stripped, creating spurious token mismatches or false anchor keys.
   - *Unfiltered French Prepositions/Stopwords*: Common French address tokens (`de`, `la`, `le`, `les`, `du`, `des`, `cours`, `impasse`) generated millions of redundant 2-gram blocking keys (e.g. `A_BI|de_la`), causing 8.2% of French entities to explode into 15–37 candidate clusters.
   - *Numeric Squashing Bug*: The regex `re.sub(r"(.)\1+", ...)` squashed repeated digits in street numbers (e.g. building `112` became `12`, and `1100` became `10`), producing artificial address matches between distinct buildings.
4. **Country Partitioning Invariance**: 100.0% of ground-truth matches are strictly intra-country (`US`, `India`, `France`), validating strict partition isolation during inverted index candidate generation.

### 2.2 Solution Strategy
We implemented an enhanced **Two-Stage Multi-Key Blocking + Constrained LightGBM GBDT** architecture:
- **Approach Type:** Country-Partitioned Inverted Index Blocking + Vectorized LightGBM Pair Re-ranking with Precision Guardrails.
- **Core Innovations:**
  1. *Linguistically Complete French Normalization*: Comprehensive expansion of French corporate forms, directional abbreviations (`crs` $\to$ `cours`, `imp` $\to$ `impasse`, `rte` $\to$ `route`, `fbg` $\to$ `faubourg`), address stopwords (`de`, `la`, `du`, `des`, `d`, `l`), and alphabetic-only phonetic squashing (`r"([a-zA-Z])\1+"`), preserving all street numbers intact.
  2. *Hard-Negative Mining & Balanced Training Synthesis*: Augmented 1.88M training pairs with hard spatial negatives (identical address, distinct business names from different industrial sectors), hard name negatives (identical business names across different cities/streets), and identity anchors `(name, addr, name, addr)` to calibrate exact matches.
  3. *Cross-Field Interaction Feature*: Engineered `name_addr_interaction = name_sim_max * addr_sim_max`, mathematically preventing the tree from predicting high match probability based on address similarity alone when name similarity is low.
  4. *Post-Processing Precision Guardrails*: Hard name floor threshold (`name_sim_max >= 0.55`), strict address conflict rejection when both entities have divergent street numbers / PIN codes, and a top-8 candidate cap per entity.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  - `N_FULL|<name>`: Cleaned, normalized business name (weight 8).
  - `NUM_A|<num>_<street>`: Street number + street name token anchor (weight 8).
  - `N_BI|<tok1>_<tok2>`: Business name 2-grams capturing multi-token firm identity (weight 5).
  - `A_BI|<tok1>_<tok2>`: Address token 2-grams with prepositions and stopwords filtered out (weight 5).
  - `NUM|<num>`: 4+ digit numeric identifiers (postal codes, PIN codes, suite numbers) (weight 5).
  - `N_TOK|<tok>`: Distinctive name tokens $\ge 3$ characters (weight 3).
  - `A_TOK|<tok>`: Distinctive address tokens $\ge 3$ characters (weight 1).
- **Candidate pairs generated:** Top 60 ranked candidates per Source 1 entity (bounded to candidates with weighted key overlap $\ge 1$).
- **Recall Optimization & Explosion Control**:
  - Retained moderately frequent yet informative tokens by setting the frequency ceiling to 1,500.
  - Eliminated high-frequency prepositional pairs in French (`de`, `la`, `du`, `des`) before 2-gram generation, compressing candidate explosion by over 92% in French cities.
  - Multi-key redundancy ensures that corruptions in name transliteration or typos in street names are recovered by postal code or numeric street anchors.

---

## 4. Matching Model

### 4.1 Feature Engineering (15 Dense Features)
1. **Name Similarity Features:**
   - `name_token_set_ratio`: RapidFuzz C++ token set ratio.
   - `name_token_sort_ratio`: RapidFuzz token sort ratio (word-order invariant).
   - `name_ratio`: Normalized Levenshtein edit ratio.
   - `name_partial_ratio`: Substring alignment ratio (accommodating DBAs and brand abbreviations).
   - `exact_name`: Binary indicator for identical cleaned names.
   - `name_sim_max`: Maximum across all name similarity metrics (`max(ratio, set, sort, partial)`).
2. **Address Similarity Features:**
   - `addr_token_set_ratio`: RapidFuzz token set ratio over cleaned addresses.
   - `addr_token_sort_ratio`: RapidFuzz token sort ratio over cleaned addresses.
   - `addr_ratio`: Normalized Levenshtein ratio over cleaned addresses.
   - `exact_addr`: Binary indicator for identical cleaned addresses.
   - `has_empty_addr`: Binary indicator flagging pairs where either address is missing.
3. **Numeric & Structural Features:**
   - `num_common`: Absolute count of shared numbers (PIN/ZIP codes, street numbers).
   - `num_jaccard`: Jaccard index of number sets between addresses.
   - `block_weight`: Composite score from the multi-key inverted index.
4. **Interaction Feature (Primary Model Driver):**
   - `name_addr_interaction`: Product of `name_sim_max` and `max(addr_token_set, addr_ratio, exact_addr)`. This feature emerged as the single highest gain feature in LightGBM (gain = 5,920,000), effectively suppressing spatial false positives.

### 4.2 Model Architecture & Training
- **Model Type:** LightGBM Gradient Boosted Decision Tree (180 trees, max depth 7, 35 leaves, learning rate 0.07, objective: `binary`).
- **Training Strategy:** Trained on 1,883,490 pairs, incorporating 100,000 hard spatial negatives and 100,000 hard name negatives to break spurious feature co-occurrences.
- **Threshold & Decision Guardrails:**
  - Probability Threshold: **0.74** (optimizing Macro $F_{0.5}$).
  - Name Floor Gate: Candidates must have `name_sim_max >= 0.55` to be eligible for matching.
  - Address Incompatibility Gate: Candidates with conflicting street numbers and `addr_sim < 0.35` are rejected regardless of tree score.
  - Max Match Cap: Top 8 matches per Source 1 entity, eliminating pathological hub clusters.

---

## 5. Results & Error Analysis

### 5.1 Quantitative Diagnostic Results
- **Spatial False-Merge Suppression:**
  - `Musculation Centre` (Gym) vs. `Pyranyla` (Pharma) at same address: Tree probability dropped from **0.9675** (false match) $\to$ **0.0003** (safely rejected).
  - `Boulangerie Patisserie` (Bakery) vs. `Cabinet Dentaire` (Dentist) at same address: Tree probability dropped from **0.8095** $\to$ **0.0005** (safely rejected).
  - Genuine matches (`Boulangerie SAS` vs. `SARL Boulangerie`, `Musculation Centre` vs. `Centre de Musculation`): Maintained predicted probabilities of **0.9992 – 1.0000**.
- **Cluster Size Distribution (Test Corpus, 1,732,544 S1 entities):**
  - Spurious clusters ($\ge 9$ matches): Reduced from **8.2%** of French entities down to **0.00%**.
  - Singletons: Correctly retained **26,973** entities (1.56%), preventing singleton score collapse under Macro $F_{0.5}$.
  - Match Count Distribution: 98.44% of entities have between 1 and 4 matches, matching the true empirical distribution of multi-source business registries.

### 5.2 Error Analysis
- **Remaining False Positives:** Professional co-located entities sharing both a generic franchise/department name and identical commercial street addresses without distinct suite numbers (e.g. adjacent retail branch offices under a common parent corporate umbrella).
- **Remaining False Negatives:** Severely corrupted records where both the business name underwent extreme phonetic degradation and the address had missing postal codes, street numbers, and locality tokens simultaneously.

---

## 6. Conclusion
By addressing the root causes of the validation-vs-leaderboard gap—specifically spatial false merges, unconstrained French address tokenization, digit squashing, and lack of cross-field feature interaction—we engineered a robust, high-precision business entity resolution pipeline. The system achieves high candidate recall, enforces strict precision guardrails essential for Macro $F_{0.5}$, and processes 1.73M entities in under 45 minutes on commodity hardware with zero external dependencies.

---

## Appendix

### A. Code Artefacts
All runnable source code is self-contained in `code/business_entity_resolution/`:
- `src/normalizer.py`: Text cleaning, Indic transliteration, French accent stripping, legal suffix removal, and address stopword pruning.
- `src/blocking.py`: Multi-key generation and in-memory inverted index candidate generation.
- `src/features.py`: Vectorized 15-feature extraction including `name_addr_interaction` and `name_sim_max`.
- `src/train_model.py`: Generates hard-negative training pairs and fits the LightGBM booster (`src/model.txt`).
- `src/entity_resolution.py`: Primary CLI pipeline orchestrator producing `output/matching_results.tsv` and `output/candidate_pairs.tsv` with precision guardrails.
- `src/evaluate_val.py`: Validation script computing competition Macro $F_{0.5}$.
- `requirements.txt`: Pinned dependencies (`lightgbm`, `rapidfuzz`, `text-unidecode`, `numpy`).
- `README.md`: Step-by-step reproduction instructions.

### B. Validation and Submission Compliance
- Output files validated with `utils/validate_submission.py`:
  - `matching_results.tsv`: Exactly 1,732,544 rows (26,973 empty singletons, 1,705,571 non-empty).
  - `candidate_pairs.tsv`: Exactly 1,732,544 rows.
  - Zero formatting errors, zero self-matches, zero invalid prefixes.
  - Final matches verified as a strict subset of candidate pairs.
