"""Vocabularies, GloVe loading and batching for the neural models (Tasks 2 and 3)."""
import random
import re
from collections import Counter

import numpy as np
import torch

PAD, UNK = 0, 1


def norm_word(w):
    """Lowercase + digits -> 0 (as in Lample et al.) for word-embedding lookup."""
    return re.sub(r"\d", "0", w.lower())


class Vocab:
    def __init__(self, train, other_splits, tags, glove_path=None, dim=100):
        self.tags = list(tags)
        self.t2i = {t: i for i, t in enumerate(self.tags)}
        counts = Counter(norm_word(w) for toks, _ in train for w in toks)
        self.train_count = counts

        # --- GloVe: keep only vectors for words that occur somewhere in the corpus
        glove = {}
        if glove_path is not None:
            needed = set(counts)
            for split in other_splits:
                needed |= {norm_word(w) for toks, _ in split for w in toks}
            with open(glove_path, encoding="utf8") as f:
                for line in f:
                    word, rest = line.split(" ", 1)
                    if word in needed:
                        glove[word] = np.asarray(rest.split(), dtype=np.float32)

        # Vocabulary = training words + GloVe words occurring in val/test.
        # (Standard practice, and what the paper does: val/test words only get their
        #  pretrained vector; no labels are used.)
        words = list(counts) + [w for w in glove if w not in counts]
        self.w2i = {"<PAD>": PAD, "<UNK>": UNK}
        for w in words:
            self.w2i.setdefault(w, len(self.w2i))

        scale = np.sqrt(3.0 / dim)
        emb = np.random.uniform(-scale, scale, (len(self.w2i), dim)).astype(np.float32)
        emb[PAD] = 0
        hit = 0
        for w, i in self.w2i.items():
            if w in glove:
                emb[i] = glove[w]
                hit += 1
        self.emb = torch.from_numpy(emb)
        self.glove_hits = hit

        chars = Counter(c for toks, _ in train for w in toks for c in w)
        self.c2i = {"<PAD>": PAD, "<UNK>": UNK}
        for c in chars:
            self.c2i.setdefault(c, len(self.c2i))

    def word_id(self, w):
        return self.w2i.get(norm_word(w), UNK)


def make_batch(sents, vocab, train=False, unk_prob=0.5, max_chars=30, device="cpu"):
    """sents: list of (tokens, tags). Returns dict of tensors."""
    B = len(sents)
    L = max(len(t) for t, _ in sents)
    words = torch.zeros(B, L, dtype=torch.long)
    tags = torch.zeros(B, L, dtype=torch.long)
    mask = torch.zeros(B, L, dtype=torch.bool)
    C = min(max(len(w) for toks, _ in sents for w in toks), max_chars)
    chars = torch.zeros(B, L, C, dtype=torch.long)
    clen = torch.ones(B, L, dtype=torch.long)
    for b, (toks, tgs) in enumerate(sents):
        for i, w in enumerate(toks):
            wid = vocab.word_id(w)
            # paper: replace singletons by UNK with prob 0.5 so <UNK> gets trained
            if train and vocab.train_count.get(norm_word(w), 0) == 1 and random.random() < unk_prob:
                wid = UNK
            words[b, i] = wid
            tags[b, i] = vocab.t2i[tgs[i]]
            mask[b, i] = True
            cs = w[:max_chars]
            clen[b, i] = len(cs)
            for j, ch in enumerate(cs):
                chars[b, i, j] = vocab.c2i.get(ch, UNK)
    lengths = mask.sum(1)
    return {k: v.to(device) for k, v in dict(words=words, chars=chars, clen=clen,
                                             tags=tags, mask=mask, lengths=lengths).items()}


def batches(data, size, shuffle):
    idx = list(range(len(data)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), size):
        yield [data[j] for j in idx[i:i + size]]
