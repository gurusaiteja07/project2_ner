"""Task 1: first-order HMM tagger.
 - MLE transitions/emissions with add-lambda smoothing (lambda = 0.1)
 - suffix (1-3 chars) + capitalization emission model for unknown words
 - uniform-over-open-class-tags fallback
 - Viterbi in log-space
"""
from collections import Counter

import numpy as np


def cap_class(w: str) -> str:
    if any(c.isdigit() for c in w):
        return "digit"
    if w.isupper() and len(w) > 1:
        return "allcap"
    if w[:1].isupper():
        return "initcap"
    if w.islower():
        return "lower"
    return "other"


def suffix(w: str, k: int) -> str:
    return w[-k:].lower()


def viterbi(E, trans, start, end):
    """Generic log-space Viterbi.
    E:     (n, T) per-position log emission scores
    trans: (T, T) log transition scores, trans[i, j] = score(i -> j)
    start: (T,)   log score of first tag;  end: (T,) log score of last tag
    Returns best tag index list. Reused by the Task 3 hybrid.
    """
    n, T = E.shape
    delta = np.empty((n, T))
    back = np.zeros((n, T), dtype=np.int64)
    delta[0] = start + E[0]
    for i in range(1, n):
        cand = delta[i - 1][:, None] + trans          # (prev, next)
        back[i] = cand.argmax(0)
        delta[i] = cand.max(0) + E[i]
    last = int((delta[-1] + end).argmax())
    path = [last]
    for i in range(n - 1, 0, -1):
        last = int(back[i, last])
        path.append(last)
    return path[::-1]


class HMMTagger:
    CAPS = ["digit", "allcap", "initcap", "lower", "other"]

    def __init__(self, tags, lam=0.1, open_tags=None):
        self.tags = list(tags)
        self.T = len(self.tags)
        self.t2i = {t: i for i, t in enumerate(self.tags)}
        self.lam = lam
        # NER: any tag can be a never-seen word -> all tags are open.
        # (For POS you would pass e.g. NOUN, PROPN, VERB, ADJ, ADV, INTJ, X.)
        self.open_tags = list(open_tags) if open_tags else self.tags
        self.open_idx = np.array([self.t2i[t] for t in self.open_tags])

    # ---------------------------------------------------------------- training
    def fit(self, sentences):
        T, lam = self.T, self.lam
        trans = np.zeros((T + 1, T + 1))      # rows: tags+START ; cols: tags+END
        emit = [Counter() for _ in range(T)]
        word_freq = Counter(w for toks, _ in sentences for w in toks)
        self.vocab = set(word_freq)

        tag_tok = np.zeros(T)
        hapax_tok = np.zeros(T)
        cap_c = [Counter() for _ in range(T)]
        sfx_c = {k: [Counter() for _ in range(T)] for k in (1, 2, 3)}

        for toks, tgs in sentences:
            prev = T                           # START
            for w, t in zip(toks, tgs):
                ti = self.t2i[t]
                trans[prev, ti] += 1
                emit[ti][w] += 1
                tag_tok[ti] += 1
                if word_freq[w] == 1:          # hapax words mimic unknown test words
                    hapax_tok[ti] += 1
                    cap_c[ti][cap_class(w)] += 1
                    for k in (1, 2, 3):
                        sfx_c[k][ti][suffix(w, k)] += 1
                prev = ti
            trans[prev, T] += 1                # END

        V = len(self.vocab)
        self.logA = np.log((trans + lam) / (trans.sum(1, keepdims=True) + lam * (T + 1)))
        # known-word emissions: P(w|t) = (C(t,w)+lam) / (C(t) + lam*(V+1))
        self.emit_cnt = emit
        self.emit_den = tag_tok + lam * (V + 1)
        # P(UNK | t): probability mass of unseen words, estimated from hapax tokens
        self.log_p_unk = np.log((hapax_tok + lam) / self.emit_den)
        self.cap_cnt, self.cap_den = cap_c, hapax_tok
        self.sfx_cnt, self.sfx_den = sfx_c, hapax_tok
        self.sfx_vocab = {k: set().union(*[set(c) for c in sfx_c[k]]) for k in (1, 2, 3)}
        # tag prior (used by Task 3 hybrid)
        self.log_prior = np.log((tag_tok + lam) / (tag_tok.sum() + lam * T))
        # views used by Viterbi / hybrid
        self.log_trans = self.logA[:T, :T]
        self.log_start = self.logA[T, :T]
        self.log_end = self.logA[:T, T]
        return self

    # ---------------------------------------------------------------- emissions
    def _oov_log_emission(self, w):
        T, lam = self.T, self.lam
        for k in (3, 2, 1):                    # back off from long to short suffix
            s = suffix(w, k)
            if s in self.sfx_vocab[k]:
                S = len(self.sfx_vocab[k])
                c = cap_class(w)
                sfx = np.array([np.log((self.sfx_cnt[k][t][s] + lam) / (self.sfx_den[t] + lam * (S + 1)))
                                for t in range(T)])
                cap = np.array([np.log((self.cap_cnt[t][c] + lam) / (self.cap_den[t] + lam * len(self.CAPS)))
                                for t in range(T)])
                return self.log_p_unk + cap + sfx
        e = np.full(T, -1e9)                   # never-seen suffix: uniform over open tags
        e[self.open_idx] = -np.log(len(self.open_idx))
        return e

    def emission_matrix(self, tokens):
        T = self.T
        E = np.empty((len(tokens), T))
        for i, w in enumerate(tokens):
            if w in self.vocab:
                E[i] = [np.log((self.emit_cnt[t][w] + self.lam) / self.emit_den[t]) for t in range(T)]
            else:
                E[i] = self._oov_log_emission(w)
        return E

    # ---------------------------------------------------------------- decoding
    def predict(self, tokens):
        if not tokens:
            return []
        E = self.emission_matrix(tokens)
        path = viterbi(E, self.log_trans, self.log_start, self.log_end)
        return [self.tags[i] for i in path]
