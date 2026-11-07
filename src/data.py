"""Loads CoNLL-2003 once, maps integer ner_tags -> strings, caches to JSON."""
import json
import random

import numpy as np

from config import CACHE_PATH, SEED


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _download():
    from datasets import load_dataset
    try:
        ds = load_dataset("eriktks/conll2003")
    except Exception as e:  # newer `datasets` versions dropped loading scripts
        print(f"[data] default load failed ({e!r}); trying parquet branch")
        ds = load_dataset("eriktks/conll2003", revision="refs/convert/parquet")
    names = ds["train"].features["ner_tags"].feature.names  # the dataset's own ClassLabel
    out = {"tag_names": names}
    for split in ("train", "validation", "test"):
        out[split] = [
            {"tokens": ex["tokens"], "tags": [names[i] for i in ex["ner_tags"]]}
            for ex in ds[split]
        ]
    return out


def load_conll():
    """Returns (splits, tag_names). splits[split] = list of (tokens, tags) with string tags."""
    if CACHE_PATH.exists():
        raw = json.loads(CACHE_PATH.read_text())
    else:
        raw = _download()
        CACHE_PATH.write_text(json.dumps(raw))
    splits = {s: [(ex["tokens"], ex["tags"]) for ex in raw[s]]
              for s in ("train", "validation", "test")}
    return splits, raw["tag_names"]
