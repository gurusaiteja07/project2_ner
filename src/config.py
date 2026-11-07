"""Shared constants and paths. Every script imports from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
GLOVE_PATH = DATA_DIR / "glove" / "glove.6B.100d.txt"
CACHE_PATH = DATA_DIR / "conll2003_cache.json"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42

# Expected CoNLL-2003 sizes (used by verify_data.py)
EXPECTED_SIZES = {"train": 14041, "validation": 3250, "test": 3453}
EXPECTED_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

# Task 1
HMM_LAMBDA = 0.1

# Task 2 (Lample et al. 2016, as given in the project spec)
NN = dict(
    char_dim=25, char_hidden=25, word_dim=100, hidden=100, dropout=0.5,
    lr=0.015, lr_decay=0.05, momentum=0.9, clip=5.0,
    max_epochs=50, patience=5, batch_size=10, singleton_unk_prob=0.5,
)
