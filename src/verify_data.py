"""Run right after downloading: checks split sizes, label set, and prints a sample."""
from collections import Counter

from config import EXPECTED_SIZES, EXPECTED_TAGS
from data import load_conll

splits, tags = load_conll()
ok = True
for s, n in EXPECTED_SIZES.items():
    got = len(splits[s])
    print(f"{s:<11} {got:>6}  (expected {n})  {'OK' if got == n else 'MISMATCH'}")
    ok &= got == n
print("tags:", tags, "OK" if sorted(tags) == sorted(EXPECTED_TAGS) else "MISMATCH")
ok &= sorted(tags) == sorted(EXPECTED_TAGS)
toks, tg = splits["train"][0]
print("sample:", list(zip(toks, tg)))
assert all(len(t) == len(g) for s in splits.values() for t, g in s), "token/tag length mismatch"
print("tag counts (train):", dict(Counter(t for _, g in splits["train"] for t in g)))
print("\nALL CHECKS PASSED" if ok else "\nSOMETHING IS OFF - fix before continuing")
