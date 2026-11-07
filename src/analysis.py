"""Compare the three models: overall + per-type F1, and recall on SEEN vs UNSEEN entities."""
import json

from seqeval.metrics import classification_report, f1_score
from seqeval.metrics.sequence_labeling import get_entities

from config import RESULTS_DIR
from data import load_conll

def per_type_f1(gold, pred):
    out = {}
    for line in classification_report(gold, pred, digits=4).splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] in ("PER", "ORG", "LOC", "MISC"):
            out[parts[0]] = float(parts[3])
    return out


FILES = {"HMM": "task1_hmm.json", "BiLSTM-CRF": "task2_bilstm_crf.json", "Hybrid": "task3_hybrid.json"}

splits, _ = load_conll()
train_ents = set()
for toks, tg in splits["train"]:
    for typ, s, e in get_entities(tg):
        train_ents.add(" ".join(toks[s:e + 1]).lower())

test = splits["test"]
rows = {}
for name, fn in FILES.items():
    p = RESULTS_DIR / fn
    if not p.exists():
        print(f"(skipping {name}: {fn} not found)")
        continue
    r = json.loads(p.read_text())
    gold, pred = r["gold"], r["pred"]
    rep = per_type_f1(gold, pred)
    seen_hit = seen_tot = uns_hit = uns_tot = 0
    for (toks, _), g, pr in zip(test, gold, pred):
        pset = set(get_entities(pr))
        for ent in get_entities(g):
            surface = " ".join(toks[ent[1]:ent[2] + 1]).lower()
            hit = ent in pset
            if surface in train_ents:
                seen_tot += 1; seen_hit += hit
            else:
                uns_tot += 1; uns_hit += hit
    rows[name] = dict(F1=f1_score(gold, pred), **{k: rep[k] for k in ("PER", "ORG", "LOC", "MISC")},
                      seen_recall=seen_hit / max(seen_tot, 1), unseen_recall=uns_hit / max(uns_tot, 1))

cols = ["F1", "PER", "ORG", "LOC", "MISC", "seen_recall", "unseen_recall"]
print(f"{'model':<12}" + "".join(f"{c:>15}" for c in cols))
for name, r in rows.items():
    print(f"{name:<12}" + "".join(f"{r[c]:>15.4f}" for c in cols))
(RESULTS_DIR / "comparison.json").write_text(json.dumps(rows, indent=2))
