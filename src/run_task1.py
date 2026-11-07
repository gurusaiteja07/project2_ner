"""Task 1: train HMM on train, (lambda sweep on validation for info), report test entity-level F1."""
import json

from seqeval.metrics import classification_report, f1_score

from config import HMM_LAMBDA, RESULTS_DIR
from data import load_conll, set_seed
from hmm import HMMTagger


def evaluate(model, sents):
    gold = [t for _, t in sents]
    pred = [model.predict(tok) for tok, _ in sents]
    return f1_score(gold, pred), gold, pred


if __name__ == "__main__":
    set_seed()
    splits, tags = load_conll()

    print("Validation lambda sweep (informational; spec fixes lambda = 0.1):")
    for lam in (0.01, 0.1, 0.5, 1.0):
        m = HMMTagger(tags, lam=lam).fit(splits["train"])
        print(f"  lambda={lam:<5} val F1 = {evaluate(m, splits['validation'])[0]:.4f}")

    model = HMMTagger(tags, lam=HMM_LAMBDA).fit(splits["train"])
    val_f1, _, _ = evaluate(model, splits["validation"])
    test_f1, gold, pred = evaluate(model, splits["test"])
    print(f"\nHMM (lambda={HMM_LAMBDA})  val F1 = {val_f1:.4f}   TEST F1 = {test_f1:.4f}")
    print(classification_report(gold, pred, digits=4))
    (RESULTS_DIR / "task1_hmm.json").write_text(json.dumps(
        {"val_f1": val_f1, "test_f1": test_f1, "pred": pred, "gold": gold}))
