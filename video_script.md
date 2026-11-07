# 🎬 Video Presentation Script: Statistical vs. Neural NER on CoNLL-2003
### NLP Assignment 2 — Individual Presentation (Total Duration: 5:00)

> **Structure & Timing Breakdown:**
> - **[0:00 – 0:30]** Problem Description *(No related works; crisp problem definition)*
> - **[0:30 – 3:00]** Code Walkthrough *(2.5 minutes; modular code breakdown across Tasks 1, 2, and 3)*
> - **[3:00 – 4:45]** Result Analysis & Seen/Unseen Generalization *(1.75 minutes; deep dive into metrics)*
> - **[4:45 – 5:00]** Conclusion & Key Takeaway *(15 seconds; final wrap-up)*

---

## [TIME: 0:00 – 0:30] — Problem Description

*[Visual: Show title slide or camera directly. Switch to a sample CoNLL-2003 sentence with BIO annotations on screen.]*

**Speaker:**
"Hello everyone. In this project, I address the problem of **Named Entity Recognition (NER)** on the standard **CoNLL-2003 English newswire dataset**, comprising over 14,000 training, 3,250 validation, and 3,453 test sentences. 

NER is framed as a token-level sequence labeling task under the **BIO scheme**, spanning 9 tags: Outside, and Begin/Inside for Person, Organization, Location, and Miscellaneous entities. 

My goal is to implement, evaluate, and contrast three distinct modeling paradigms: a classical statistical **first-order HMM**, a neural **BiLSTM-CRF**, and a **Hybrid model** blending neural emissions with statistical transition priors—all strictly evaluated using entity-level micro F1."

---

## [TIME: 0:30 – 3:00] — Code Walkthrough (Tasks 1, 2, & 3)

*[Visual: Switch screen to IDE / VS Code showing the `project2_ner/src/` folder tree, highlighting key files as you mention them.]*

**Speaker:**
"Let's walk through the codebase in `project2_ner/src`, which is structured into modular, reproducible components.

*[Visual: Open `src/hmm.py` and scroll to `fit()`, `_oov_log_emission()`, and `viterbi()`]*

Starting with **Task 1 in `hmm.py`**:
I implemented the `HMMTagger` from scratch. The `fit()` method computes transition bigrams $A$ and emission counts $B$ using maximum likelihood estimation with add-0.1 smoothing. 

The biggest challenge in statistical NER is handling out-of-vocabulary words. To solve this, in `_oov_log_emission()`, I extract character suffix features—backing off from 3 down to 1 character—combined with capitalization categories like `allcap`, `initcap`, and `digit`. These distributions are estimated exclusively from *hapax legomena*—words appearing only once in training—with a uniform fallback over open tags. 

Inference is driven by `viterbi()`, a standalone, log-space dynamic programming implementation that avoids numerical underflow and returns the globally optimal tag sequence.

*[Visual: Open `src/nn_data.py`, then `src/model.py` and `src/crf.py`]*

Moving to **Task 2—the neural architecture**:
In `nn_data.py`, the `Vocab` class loads 100-dimensional pretrained GloVe vectors for training tokens, plus validation and test tokens present in GloVe, yielding over 20,000 vector hits with zero label leakage. During batch creation, singleton training words are randomly replaced with `UNK` at probability 0.5 to train the unknown embedding, and all digit sequences are normalized to zeros.

In `model.py`, the network uses two hierarchical representation layers:
First, a character-level BiLSTM encodes character sequences into 50-dimensional subword vectors, capturing prefixes, suffixes, and casing.
Second, this vector is concatenated with the 100-dimensional GloVe word embedding to form a 150-dimensional input, passed through 50% dropout into a bidirectional word LSTM with 100 hidden units per direction, outputting a 200-dimensional contextualized representation.

Rather than using an external library, I implemented the linear-chain CRF from scratch in `crf.py`. It computes the negative log-likelihood via a vectorized forward algorithm using `logsumexp` to compute the partition function $Z(x)$, and uses Viterbi decoding with transition parameters $A_{i,j}$ to enforce globally valid BIO transitions. In `train_utils.py`, the model is trained with SGD, momentum 0.9, gradient clipping at 5.0, and an epoch-wise learning rate decay schedule.

*[Visual: Open `src/run_task3.py` and highlight `decode_all()` and the grid search loop]*

Finally, for **Task 3 in `run_task3.py`**:
I built a hybrid decoder. It replaces the HMM's discrete count emission matrix with contextual log-probabilities from a trained BiLSTM-Softmax model. 

The emission score subtracts $\alpha$ times the unigram tag prior to prevent dominant tags like 'O' from taking over, and adds $\beta$ times the HMM transition log-probabilities. I ran a validation grid search over 30 combinations of $\alpha$ and $\beta$, evaluating on the test set strictly once using our modular `viterbi()` function."

---

## [TIME: 3:00 – 4:45] — Result Analysis & Generalization

*[Visual: Switch to presentation slide showing the Full Results Table and Seen vs. Unseen Recall charts.]*

**Speaker:**
"Now let's examine the experimental results on the test set.

Here is the comparative summary:
- The **HMM baseline** achieves an overall F1 of **72.10%**.
- The **BiLSTM-CRF** achieves an overall F1 of **90.84%**—a massive **+18.74 percentage point leap**.
- The **Hybrid model** scores **90.32%**, closely matching the neural system.

*[Visual: Highlight the Seen vs. Unseen Recall rows in the table]*

The most illuminating finding comes from analyzing **Seen versus Unseen entity recall**:
On entities seen during training:
The HMM achieves a respectable **87.90%** recall, while the BiLSTM-CRF achieves **94.57%**. 

However, on **unseen entities** never encountered in training:
The HMM's recall collapses to just **50.28%**—a 37.6 percentage point drop. In contrast, the BiLSTM-CRF maintains an outstanding **85.68%** unseen recall—a **+35.40 percentage point improvement** over the HMM. 

Why is this difference so profound? 
The HMM relies on exact discrete word lookups; its suffix heuristic helps, but has no semantic understanding. The neural model, however, leverages GloVe's continuous vector space where unseen names like foreign locations or new figures automatically cluster near known entities. Concurrently, the character BiLSTM directly extracts morphological features, allowing the neural model to generalize effortlessly to novel entities.

*[Visual: Highlight the PER, LOC, ORG, MISC column breakdowns]*

Looking at individual entity types:
- **PER** was easiest across all models, peaking at **95.48%** in the BiLSTM-CRF due to regular capitalization patterns.
- **LOC** followed closely at **92.82%**.
- **ORG** was the hardest category for all systems, scoring only **64.60%** on the HMM and **88.86%** on the neural model, driven by naming ambiguity and overlap with common dictionary words.
- Interestingly, the **Hybrid model** slightly edged out the BiLSTM-CRF on **MISC entities**—**80.43%** versus 80.23%—showing that statistical transition priors provide a beneficial regularizing constraint on formulaic miscellaneous phrasing."

---

## [TIME: 4:45 – 5:00] — Conclusion & Key Takeaways

*[Visual: Switch back to camera or show conclusion slide with 3 bullet takeaways.]*

**Speaker:**
"To conclude: while statistical HMMs provide an interpretable baseline, their discrete nature imposes a fatal OOV bottleneck. The BiLSTM-CRF, equipped with pretrained embeddings, subword character representations, and joint sequence decoding, decisively solves this generalization gap, delivering robust, high-precision sequence labeling. 

Thank you for your time!"

---

### ⏱️ Pacing & Recording Guide for Students
- **Word count:** ~725 spoken words.
- **Pacing:** Normal conversational speaking pace is ~145–150 words per minute.
- **Checkpoints:**
  - Reach the code walkthrough by **0:30**.
  - Transition to the results table around **3:00**.
  - Begin the conclusion slide at **4:45**.
