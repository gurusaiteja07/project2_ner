# Statistical vs. Neural Named Entity Recognition on CoNLL-2003
**Course:** Natural Language Processing (Assignment 2)  
**Evaluation Metric:** Entity-level Micro F1 (`seqeval`, strict boundary & type match)

---

## 1. Problem Statement & Dataset

Named Entity Recognition (NER) is a sequence labeling task that detects and classifies contiguous token spans into semantic categories: **Person (PER)**, **Organization (ORG)**, **Location (LOC)**, and **Miscellaneous (MISC)**.

We evaluate three sequence labeling paradigms on the standard **CoNLL-2003 English newswire dataset** under the standard **BIO scheme** (9 tags: `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`, `B-MISC`, `I-MISC`).

| Split | Sentences | Tokens | Role |
|---|---|---|---|
| **Train** | 14,041 | ~203k | Parameter estimation / Model training |
| **Validation** | 3,250 | ~51k | Hyperparameter tuning & early stopping |
| **Test** | 3,453 | ~46k | Final unbiased evaluation |

*Note on Class Distribution:* Approximately 83% of all tokens are `O`. Strict entity-level F1 is required to penalize trivial majority-class predictions.

---

## 2. Methodology & Architecture

### Task 1: First-Order Hidden Markov Model (Statistical Baseline)
- **Generative Formulation:** Models joint probability $P(\mathbf{w}, \mathbf{t}) = \prod_i P(t_i \mid t_{i-1}) P(w_i \mid t_i)$.
- **Parameter Estimation:** Maximum likelihood bigram transitions $A$ and emissions $B$ with add-$\lambda$ ($\lambda = 0.1$) Laplace smoothing.
- **OOV Handling:** Out-of-vocabulary words use a suffix-based back-off model (3 $\to$ 2 $\to$ 1 characters) combined with capitalization class (`allcap`, `initcap`, `digit`, `lower`), estimated from *hapax legomena* (words with training frequency = 1). Unseen suffixes fall back to uniform log-probability over open tags.
- **Decoding:** Exact dynamic programming via Viterbi in log-space.

### Task 2: BiLSTM-CRF (Neural Sequence Labeler)
Following Lample et al. (2016), combining subword morphology, distributed semantics, and structured sequence decoding:
1. **Character BiLSTM:** Character embeddings (25d) fed into a character-level BiLSTM (hidden=25) $\to$ 50d morphology vector.
2. **Word Representations:** 100d pre-trained GloVe vectors (`glove.6B.100d.txt`) fine-tuned during training. Singletons are stochastically replaced with `<UNK>` with probability $p = 0.5$ during training; digits normalized to `0`.
3. **Sentence Encoder:** Fused input (150d) $\to$ Dropout (0.5) $\to$ Bidirectional LSTM (hidden=100) $\to$ 200d contextual representation.
4. **Custom CRF Layer (`crf.py`):** Hand-crafted linear-chain CRF without external packages. Computes partition function $Z(\mathbf{x})$ via vectorized forward algorithm with `logsumexp`, trained with Negative Log-Likelihood, and decoded with Viterbi.
5. **Training Setup:** SGD with momentum (0.9), gradient clipping (5.0), learning rate decay $\eta_t = 0.015 / (1 + 0.05t)$, mini-batch size 10 (summed loss), and early stopping (patience=5).

### Task 3: Hybrid Sequence Labeler
Explores whether count-based HMM transition constraints can complement neural emissions:
- **Neural Emission:** Per-token log-probabilities from a trained BiLSTM-Softmax model. To prevent dominant classes (`O`) from skewing paths, the empirical log tag prior is scaled and subtracted:
  $$\text{Emission}_i(t) = \log P_{\text{nn}}(t \mid \mathbf{w}, i) - \alpha \cdot \log P(t)$$
- **Scoring Function:** Total path score blends scaled neural emissions with HMM transition log-probabilities:
  $$\text{Score}(\mathbf{t}) = \sum_{i=1}^n \left[ \text{Emission}_i(t_i) + \beta \cdot \log P_{\text{HMM}}(t_i \mid t_{i-1}) \right]$$
- **Tuning:** $(\alpha, \beta)$ tuned via exhaustive grid search on validation set across $\alpha \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$ and $\beta \in \{0.0, 0.25, 0.5, 1.0, 1.5, 2.0\}$ (30 combinations), evaluated on test once.

---

## 3. Results & Performance Comparison

All models were evaluated on the CoNLL-2003 Test set:

| Model | Overall F1 | PER F1 | LOC F1 | ORG F1 | MISC F1 | Seen Recall | Unseen Recall |
|---|---|---|---|---|---|---|---|
| **Task 1: HMM** | 0.7210 | 0.7465 | 0.7726 | 0.6460 | 0.7007 | 0.8790 | 0.5028 |
| **Task 2: BiLSTM-CRF** | **0.9084** | **0.9548** | **0.9282** | **0.8886** | 0.8023 | **0.9457** | **0.8568** |
| **Task 3: Hybrid** | 0.9032 | 0.9498 | 0.9215 | 0.8805 | **0.8043** | 0.9444 | 0.8468 |

---

## 4. Key Findings & Discussion

### 1. The OOV Bottleneck (Seen vs. Unseen Recall)
- **HMM Performance Gap:** The HMM achieves strong recall on seen entities (**87.90%**), but collapses to **50.28%** on unseen words—a **37.62 pp drop**. Discrete lexical counting cannot assign informative emissions to novel vocabulary, even with suffix heuristics.
- **Neural Generalization:** The BiLSTM-CRF achieves **85.68%** recall on unseen entities (**+35.40 pp over HMM**). Pre-trained GloVe embeddings map unseen entity words close to known semantic clusters, while the Char-BiLSTM extracts morphological signals (casing, prefixes, suffixes) automatically.

### 2. Entity-Type Difficulty Analysis
- **PER is Easiest (95.48% F1):** Strong capitalization regularity and localized syntactic triggers (titles, honorifics) make person names easiest for all models.
- **ORG is Hardest (88.86% Neural, 64.60% HMM):** High lexical diversity, multi-word spans, and frequent semantic overlap with common dictionary words or locations cause boundary and classification errors.
- **MISC Edge in Hybrid (80.43% vs 80.23%):** Miscellaneous entities (nationalities, events) follow formulaic bigram syntactic patterns where the static HMM transition prior provides a beneficial regularizing constraint.

### 3. Role of the Hybrid Model
- Tuning converged to small values of $\alpha$ and $\beta$, showing that contextual BiLSTM emission scores already provide well-calibrated decisions.
- While the hybrid proves that statistical transition tables can replace a learned CRF with minimal performance loss (-0.52 pp), the end-to-end trained CRF remains superior because transition weights are optimized jointly with the feature encoder.

---

## 5. Reproduction & Code Structure

All tasks are executed from `project2_ner/src`:

| Script | Purpose | Execution Command | Output Artifact |
|---|---|---|---|
| `run_task1.py` | Trains HMM & sweeps $\lambda$ | `python run_task1.py` | `results/task1_hmm.json` |
| `run_task2.py` | Trains BiLSTM-CRF | `python run_task2.py` | `results/task2_bilstm_crf.pt` |
| `run_task3.py` | Runs $(\alpha, \beta)$ grid search & tests hybrid | `python run_task3.py` | `results/task3_hybrid.json` |
| `analysis.py` | Generates comparison table & seen/unseen recall | `python analysis.py` | `results/comparison.json` |

Unit tests verify correctness across all components (`python -m pytest tests -q` $\to$ 11/11 passed).
