"""Task 3: hybrid = HMM transition table + Viterbi decoder, with the counted emission table
replaced by the BiLSTM's learned scores.

  emission_i(t) = log P_nn(t | sentence, i) - alpha * log P(t)      (scaled likelihood)
  score(path)   = sum_i [ emission_i(t_i) + beta * log P_hmm(t_i | t_{i-1}) ]

alpha, beta are chosen on VALIDATION only; test is evaluated once.
--source softmax : train a fresh BiLSTM with a softmax output (no CRF)   [default, recommended]
--source crf     : reuse Task 2's trained BiLSTM-CRF, log_softmax over its emission scores
"""
import argparse
import itertools
import json

import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score

from config import GLOVE_PATH, NN, RESULTS_DIR
from data import load_conll, set_seed
from hmm import HMMTagger, viterbi
from model import BiLSTMTagger
from nn_data import Vocab, batches, make_batch
from train_utils import evaluate, train


def nn_logprobs(model, sents, vocab, device):
    model.eval()
    out = []
    for chunk in batches(sents, 64, shuffle=False):
        out += model.log_probs(make_batch(chunk, vocab, device=device))
    return out


def decode_all(lp, hmm, alpha, beta):
    preds = []
    for P in lp:
        E = P - alpha * hmm.log_prior
        path = viterbi(E, beta * hmm.log_trans, beta * hmm.log_start, beta * hmm.log_end)
        preds.append([hmm.tags[i] for i in path])
    return preds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["softmax", "crf"], default="softmax")
    args = ap.parse_args()

    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits, tags = load_conll()
    vocab = Vocab(splits["train"], [splits["validation"], splits["test"]], tags, GLOVE_PATH, NN["word_dim"])
    hmm = HMMTagger(tags).fit(splits["train"])       # only its transition table + tag prior are used

    use_crf = args.source == "crf"
    model = BiLSTMTagger(len(vocab.c2i), len(tags), vocab.emb, NN["char_dim"], NN["char_hidden"],
                         NN["hidden"], NN["dropout"], use_crf=use_crf).to(device)
    if use_crf:
        model.load_state_dict(torch.load(RESULTS_DIR / "task2_bilstm_crf.pt", map_location=device))
    else:
        train(model, splits["train"], splits["validation"], vocab, NN, device,
              RESULTS_DIR / "task3_bilstm_softmax.pt")

    gold_val = [t for _, t in splits["validation"]]
    gold_test = [t for _, t in splits["test"]]
    lp_val = nn_logprobs(model, splits["validation"], vocab, device)
    lp_test = nn_logprobs(model, splits["test"], vocab, device)

    grid = []
    for alpha, beta in itertools.product([0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]):
        f1 = f1_score(gold_val, decode_all(lp_val, hmm, alpha, beta))
        grid.append((f1, alpha, beta))
        print(f"alpha={alpha:<4} beta={beta:<4} val F1={f1:.4f}")
    val_f1, alpha, beta = max(grid)
    print(f"\nbest on validation: alpha={alpha} beta={beta} val F1={val_f1:.4f}")

    pred = decode_all(lp_test, hmm, alpha, beta)
    test_f1 = f1_score(gold_test, pred)
    nn_only = f1_score(gold_test, decode_all(lp_test, hmm, 0.0, 0.0))   # pure per-word argmax
    print(f"Hybrid TEST F1 = {test_f1:.4f}   (BiLSTM per-word argmax, no transitions: {nn_only:.4f})")
    print(classification_report(gold_test, pred, digits=4))
    (RESULTS_DIR / "task3_hybrid.json").write_text(json.dumps(
        {"source": args.source, "alpha": alpha, "beta": beta, "val_f1": val_f1, "test_f1": test_f1,
         "nn_argmax_test_f1": nn_only, "pred": pred, "gold": gold_test}))
