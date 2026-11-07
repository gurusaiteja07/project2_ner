"""Task 2: reproduce Lample et al. (2016) char+word BiLSTM-CRF on CoNLL-2003."""
import json
import sys

import torch
from seqeval.metrics import classification_report

from config import GLOVE_PATH, NN, RESULTS_DIR
from data import load_conll, set_seed
from model import BiLSTMTagger
from nn_data import Vocab
from train_utils import evaluate, train

if __name__ == "__main__":
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not GLOVE_PATH.exists():
        sys.exit(f"GloVe file missing: {GLOVE_PATH}  (see README step 3)")
    splits, tags = load_conll()
    vocab = Vocab(splits["train"], [splits["validation"], splits["test"]], tags, GLOVE_PATH, NN["word_dim"])
    print(f"device={device}  words={len(vocab.w2i)}  chars={len(vocab.c2i)}  glove hits={vocab.glove_hits}")

    model = BiLSTMTagger(len(vocab.c2i), len(tags), vocab.emb, NN["char_dim"], NN["char_hidden"],
                         NN["hidden"], NN["dropout"], use_crf=True).to(device)
    best_val, hist = train(model, splits["train"], splits["validation"], vocab, NN, device,
                           RESULTS_DIR / "task2_bilstm_crf.pt")
    test_f1, gold, pred = evaluate(model, splits["test"], vocab, device)
    print(f"\nBiLSTM-CRF  best val F1 = {best_val:.4f}   TEST F1 = {test_f1:.4f}")
    print(classification_report(gold, pred, digits=4))
    (RESULTS_DIR / "task2_bilstm_crf.json").write_text(json.dumps(
        {"val_f1": best_val, "test_f1": test_f1, "history": hist, "pred": pred, "gold": gold}))
