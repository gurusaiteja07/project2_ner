# Statistical vs. Neural NER on CoNLL-2003

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-CoNLL--2003-orange)
![GloVe](https://img.shields.io/badge/Embeddings-GloVe%20100d-purple)

A from-scratch implementation comparing **statistical** (HMM) and **neural** (BiLSTM-CRF) approaches to Named Entity Recognition on the CoNLL-2003 English benchmark, plus a novel **hybrid** model that fuses both. Built as NLP Assignment 2.

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Requirements](#setup--requirements)
- [Data Setup](#data-setup)
- [Running the Code](#running-the-code)
- [Key Implementation Details](#key-implementation-details)
- [Analysis & Findings](#analysis--findings)
- [References](#references)
- [License](#license)

---

## Overview

This project implements and compares three NER systems across three progressive tasks:

| Task | Model | Description |
|------|-------|-------------|
| **Task 1** | HMM Tagger | First-order Hidden Markov Model with add-λ smoothing, suffix + capitalisation OOV model, and log-space Viterbi decoding |
| **Task 2** | BiLSTM-CRF | Character + word-level Bidirectional LSTM with a custom linear-chain CRF output layer, GloVe 100d initialisation (Lample et al., 2016) |
| **Task 3** | Hybrid | Combines BiLSTM emission scores with HMM transition structure; neural emissions replace HMM emissions while reusing the trained HMM transition matrix through shared Viterbi decoding |

---

## Results

Performance on the **CoNLL-2003 test set** (entity-level F1, `seqeval`):

| Model | F1 | PER | ORG | LOC | MISC | Seen Recall | Unseen Recall |
|---|---|---|---|---|---|---|---|
| HMM | 0.7210 | 0.7465 | 0.6460 | 0.7726 | 0.7007 | 0.8790 | 0.5028 |
| BiLSTM-CRF | **0.9084** | **0.9548** | **0.8886** | **0.9282** | 0.8023 | **0.9457** | **0.8568** |
| Hybrid | 0.9032 | 0.9498 | 0.8805 | 0.9215 | **0.8043** | 0.9444 | 0.8468 |

> **Key takeaway:** The BiLSTM-CRF achieves a **+18.7 point** F1 gain over the HMM. The Hybrid model matches the BiLSTM-CRF almost exactly while achieving the **highest MISC F1** — the category the neural model struggles with most — suggesting that HMM transitions provide a useful inductive bias for rare entity types.

---

## Architecture

### Task 1 — Hidden Markov Model

```
Input tokens:  [EU]  [rejects]  [German]  [call]  ...
                |        |          |        |
            Emission  Emission  Emission  Emission     <- P(w | tag)
            P(w|t)    P(w|t)    P(w|t)    P(w|t)
                \        |          |       /
                 ----Transition Model------            <- P(tag_i | tag_{i-1})
                       Viterbi                         <- log-space, O(n·T²)
                 -------------------------
Output tags:  [B-ORG]   [O]    [B-MISC]   [O]  ...

OOV model (unknown words):
  word → suffix(1–3 chars) + cap_class → back-off emission
  cap classes: digit | allcap | initcap | lower | other
```

### Task 2 — BiLSTM-CRF

```
                     ┌──────────────────────────────────────────┐
                     │           CHARACTER ENCODER               │
  "rejects" → [r,e,j,e,c,t,s] → CharEmb(25d) → BiLSTM → [h_fwd ⊕ h_bwd] (50d)
                     └──────────────┬───────────────────────────┘
                                    │
  GloVe(100d) → WordEmb(100d) ─────┤ concat (150d)
                                    │
                              Dropout(0.5)
                                    │
                        ┌───────────▼──────────────┐
                        │  Sentence BiLSTM (100d)  │
                        │  fwd ──────────────────► │
                        │  bwd ◄────────────────── │
                        └───────────┬──────────────┘
                                    │ 200d
                              Dropout(0.5)
                                    │
                           Linear → emissions (n_tags)
                                    │
                        ┌───────────▼──────────────┐
                        │   Linear-Chain CRF        │
                        │  Viterbi decode           │
                        └──────────────────────────┘
Output: [B-ORG, O, B-MISC, O, ...]
```

### Task 3 — Hybrid (Neural Emissions + HMM Transitions)

```
BiLSTM (softmax head) ──► log P_nn(tag | sentence)   ← per-token neural log-probs
                                    │
       emission = log P_nn(t|x) − α · log P_hmm(t)   ← subtract HMM tag prior
                                    │
HMM trained transitions ──────────► log P(t_i | t_{i-1})
                                    │
                           Shared Viterbi decoder      ← same log-space Viterbi from Task 1
                                    │
                           α, β grid-searched on val   ← validation-only tuning
Output: [B-ORG, O, B-MISC, O, ...]
```

---

## Project Structure

```
project2_ner/
├── README.md                        ← You are here
├── requirements.txt                 ← pip dependencies
│
├── data/
│   ├── conll2003_cache.json         ← auto-created on first run of verify_data.py
│   └── glove/
│       └── glove.6B.100d.txt        ← download manually (see Data Setup)
│
├── src/
│   ├── config.py          Shared paths + all hyperparameters (single source of truth)
│   ├── data.py            CoNLL-2003 loader, tag ↔ int conversion, seed utilities
│   ├── hmm.py             Task 1: HMMTagger + reusable log-space Viterbi
│   ├── run_task1.py       Train + evaluate HMM, writes results/task1_hmm.json
│   ├── nn_data.py         Vocabulary, GloVe loading, char IDs, batch collation
│   ├── crf.py             Custom linear-chain CRF (NLL + Viterbi)
│   ├── model.py           Char+word BiLSTM (use_crf=True/False)
│   ├── train_utils.py     SGD loop, lr decay, gradient clipping, early stopping
│   ├── run_task2.py       Train + evaluate BiLSTM-CRF, writes results/task2_bilstm_crf.json
│   ├── run_task3.py       Grid-search hybrid (α, β), writes results/task3_hybrid.json
│   ├── verify_data.py     Downloads + validates CoNLL-2003; sanity-checks tag set & sizes
│   └── analysis.py        Loads all results, prints comparison table, writes results/analysis.txt
│
├── tests/
│   ├── conftest.py        Shared fixtures (tiny toy sentences, mock vocab)
│   ├── test_hmm.py        Unit tests: HMM training, Viterbi, OOV model (7 tests)
│   ├── test_crf.py        Unit tests: CRF forward/backward, NLL, Viterbi
│   └── test_model.py      Unit tests: BiLSTM forward pass, emission shapes
│
└── results/               Created automatically; git-ignored for large .pt files
    ├── task1_hmm.json
    ├── task2_bilstm_crf.json
    ├── task2_bilstm_crf.pt          ← trained model checkpoint
    ├── task3_hybrid.json
    ├── task3_hybrid_crf.json
    ├── comparison.json
    └── analysis.txt
```

---

## Setup & Requirements

**Python 3.10 or higher** is required (tested on 3.13 and 3.14).

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
torch>=2.0
datasets>=2.14
seqeval>=1.2.2
numpy
tqdm
pytest
```

> **Tip:** A GPU is not required but significantly speeds up Tasks 2 and 3. On CPU, expect ~5–10 min per training epoch.

---

## Data Setup

### CoNLL-2003 (automatic)

CoNLL-2003 is downloaded and cached automatically via HuggingFace `datasets` on the first run of `verify_data.py`. No manual step needed.

```
Expected splits: train=14041 | val=3250 | test=3453 sentences
Tags: O  B-PER  I-PER  B-ORG  I-ORG  B-LOC  I-LOC  B-MISC  I-MISC
```

The raw dataset is cached to `data/conll2003_cache.json` for fast offline re-runs.

### GloVe 100d (required for Tasks 2 & 3)

Download and extract GloVe into `data/glove/`:

```bash
# From the project root (project2_ner/)
mkdir -p data/glove
cd data/glove

# Option A — wget (Linux/macOS/WSL)
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.100d.txt
rm glove.6B.zip

# Option B — HuggingFace Hub mirror (if Stanford is slow)
# pip install huggingface_hub
# python -c "from huggingface_hub import hf_hub_download; hf_hub_download('stanfordnlp/glove', 'glove.6B.zip', local_dir='.')"
# unzip glove.6B.zip glove.6B.100d.txt
```

| File | Size |
|------|------|
| `glove.6B.zip` | ~822 MB |
| `glove.6B.100d.txt` (extracted) | ~347 MB |

> **Note:** Only the 100d file is used. You do **not** need the 50d, 200d, or 300d variants.

---

## Running the Code

> **All scripts must be run from the `src/` directory.**

```bash
cd project2_ner/src
```

### Step 0 — Run Unit Tests

```bash
cd ..          # project2_ner/
pytest tests -q
cd src
```

**Expected output:** All tests pass (HMM Viterbi, CRF forward/backward, BiLSTM shape checks).

---

### Step 1 — Verify Data

```bash
python verify_data.py
```

**Expected output:**
```
train 14041 OK | validation 3250 OK | test 3453 OK
Tags (9): ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'] OK
First sentence: EU B-ORG | rejects O | German B-MISC | call O | to O | boycott O | British B-MISC | lamb O | . O
```

---

### Step 2 — Task 1: Train & Evaluate HMM

```bash
python run_task1.py
```

**Expected output:** Test F1 ≈ 0.70–0.73; PER and LOC higher than ORG and MISC.

Writes: `results/task1_hmm.json`

---

### Step 3 — Download GloVe

*(Skip if already done — see [Data Setup](#data-setup) above.)*

```bash
# Verify the file is present and sized correctly (~347 MB)
# Windows PowerShell:
Get-Item ..\data\glove\glove.6B.100d.txt | Select-Object Length
```

---

### Step 4 — Task 2: Train & Evaluate BiLSTM-CRF

```bash
python run_task2.py
```

**Expected output:**
```
GloVe hits: 23xxx / 26xxx vocab words
Epoch 1 | loss ... | val F1 0.70–0.85
...
Epoch N | val F1 ~0.90  [early stop]
Test F1: ~0.90–0.91
```

Writes: `results/task2_bilstm_crf.json`, `results/task2_bilstm_crf.pt`

> Training typically converges within 15–25 epochs with early stopping (patience=5).

---

### Step 5 — Task 3: Train & Evaluate Hybrid

```bash
python run_task3.py
```

**Expected output:** Grid of validation F1 over (α, β), best config printed, then test F1.

Writes: `results/task3_hybrid.json`, `results/task3_hybrid_crf.json`

> **Note:** The hybrid reuses the BiLSTM checkpoint from Task 2 and the HMM from Task 1. Run Steps 2 and 4 first.

---

### Step 6 — Print Comparison Analysis

```bash
python analysis.py
```

**Expected output:**

```
model            F1      PER      ORG      LOC     MISC  seen_recall  unseen_recall
HMM          0.7210   0.7465   0.6460   0.7726   0.7007       0.8790         0.5028
BiLSTM-CRF   0.9084   0.9548   0.8886   0.9282   0.8023       0.9457         0.8568
Hybrid       0.9032   0.9498   0.8805   0.9215   0.8043       0.9444         0.8468
```

Writes: `results/analysis.txt`, `results/comparison.json`

---

## Key Implementation Details

### HMM (Task 1)

| Hyperparameter | Value | Rationale |
|---|---|---|
| Smoothing λ | 0.1 | Add-λ for transitions and emissions |
| OOV model | suffix(1–3 chars) + cap_class | Back-off from longest to shortest seen suffix |
| Open-class tags | all 9 tags | NER allows any tag for unseen words |
| Decoding | Log-space Viterbi | O(n·T²) = O(n·81) |

**OOV back-off order:** suffix-3 → suffix-2 → suffix-1 → uniform over open tags.  
**Hapax estimation:** OOV feature weights estimated exclusively from count-1 training words, which approximate the OOV distribution.

### BiLSTM-CRF (Task 2) — following Lample et al. (2016)

| Hyperparameter | Value |
|---|---|
| Character embedding dim | 25 |
| Character BiLSTM hidden | 25 (→ 50d concat) |
| Word embedding dim | 100 (GloVe 100d, fine-tuned) |
| Sentence BiLSTM hidden | 100 (→ 200d concat) |
| Dropout | 0.5 (applied before both LSTMs) |
| Optimiser | SGD + momentum 0.9 |
| Initial learning rate | 0.015 |
| LR schedule | `lr_t = 0.015 / (1 + 0.05·t)` per epoch |
| Gradient clipping | max norm 5.0 |
| Batch size | 10 sentences (summed loss) |
| Max epochs / patience | 50 / 5 |
| Singleton UNK prob | 0.5 (training only) |
| Vocabulary | Train words ∪ GloVe words appearing in val/test |

**Digit normalisation:** All digit strings are replaced with `0` for word lookup (e.g. `1996` → `0`).  
**Lowercase word lookup:** Word embeddings use lowercase tokens; character features keep the original casing.

### Hybrid (Task 3)

The hybrid replaces the HMM emission term with neural log-probabilities while keeping HMM transitions:

$$\text{emission}(t, x_i) = \log P_{\text{nn}}(t \mid \mathbf{x}) - \alpha \cdot \log P_{\text{hmm}}(t)$$

$$\text{score}(t_1 \dots t_n) = \sum_i \text{emission}(t_i, x_i) + \beta \cdot \sum_i \log P_{\text{hmm}}(t_i \mid t_{i-1})$$

- **α** subtracts the HMM tag prior to prevent double-counting.
- **β** scales the HMM transition contribution.
- Both are grid-searched on the validation set; test is touched once.

---

## Analysis & Findings

### 1. Statistical vs. Neural Gap

The BiLSTM-CRF outperforms the HMM by **+18.7 F1 points** overall. The gap is most dramatic on **unseen entities** (0.5028 → 0.8568, +35.4 points), where the neural model's character-level representations generalise far better than the HMM's suffix + capitalisation features.

### 2. Entity-Type Breakdown

| Observation | Explanation |
|---|---|
| PER is easiest for both models | Capitalisation is a very strong signal; names cluster well |
| ORG is hardest for HMM | Organisation names are diverse, lowercase, and context-dependent |
| MISC is relatively harder for BiLSTM-CRF | Small class size; rare and highly varied surface forms |
| Hybrid wins on MISC (+0.002) | HMM transition structure regularises the MISC boundary prediction |

### 3. Seen vs. Unseen Entity Recall

The HMM's seen-entity recall (87.9%) is surprisingly competitive — the add-λ smoothing and transition model work well when surface forms have been observed. The critical failure mode is **unseen entities**: the OOV model achieves only 50.3% recall, confirming that suffix heuristics cannot substitute for context-sensitive representations.

### 4. Hybrid Model: When Does Structure Help?

The hybrid does not improve over the BiLSTM-CRF on overall F1 (-0.52 points) because the BiLSTM-CRF already learns better transition potentials end-to-end via the CRF layer. However, the hybrid:
- **Improves MISC** (0.8043 vs. 0.8023) — HMM transitions provide regularisation for rare types.
- **Matches BiLSTM-CRF on seen recall** (0.9444 vs. 0.9457) within noise.
- Demonstrates that classical structure can complement neural emissions even at scale.

---

## References

1. **Lample et al. (2016).** Neural Architectures for Named Entity Recognition. *NAACL-HLT 2016*. [[PDF](https://arxiv.org/abs/1603.01360)]
2. **Tjong Kim Sang & De Meulder (2003).** Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition. *CoNLL 2003*. [[ACL Anthology](https://aclanthology.org/W03-0419/)]
3. **Pennington et al. (2014).** GloVe: Global Vectors for Word Representation. *EMNLP 2014*. [[PDF](https://nlp.stanford.edu/pubs/glove.pdf)]
4. **Rabiner (1989).** A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition. *Proceedings of the IEEE*.

---

## License

This project is licensed under the **MIT License** — see below.

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
