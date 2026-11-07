import itertools

import numpy as np
from hmm import HMMTagger, cap_class, viterbi

TAGS = ["O", "B-PER", "I-PER", "B-LOC"]
TOY = [
    (["John", "Smith", "visited", "Paris", "."], ["B-PER", "I-PER", "O", "B-LOC", "O"]),
    (["Mary", "Jones", "visited", "London", "."], ["B-PER", "I-PER", "O", "B-LOC", "O"]),
    (["Anna", "Brown", "loves", "Rome", "."], ["B-PER", "I-PER", "O", "B-LOC", "O"]),
] * 5


def test_transition_rows_sum_to_one():
    m = HMMTagger(TAGS, lam=0.1).fit(TOY)
    assert np.allclose(np.exp(m.logA).sum(1), 1.0)


def test_smoothing_gives_no_zero_probability():
    m = HMMTagger(TAGS, lam=0.1).fit(TOY)
    assert np.isfinite(m.logA).all()                       # unseen transitions still > 0
    assert np.isfinite(m.emission_matrix(["John"])).all()


def test_viterbi_equals_brute_force():
    rng = np.random.default_rng(0)
    n, T = 5, 4
    E, tr = rng.normal(size=(n, T)), rng.normal(size=(T, T))
    st, en = rng.normal(size=T), rng.normal(size=T)
    best = max(itertools.product(range(T), repeat=n),
               key=lambda p: st[p[0]] + E[0, p[0]] + sum(tr[p[i-1], p[i]] + E[i, p[i]] for i in range(1, n)) + en[p[-1]])
    assert viterbi(E, tr, st, en) == list(best)


def test_tags_training_set_nearly_perfectly():
    m = HMMTagger(TAGS).fit(TOY)
    for toks, gold in TOY[:3]:
        assert m.predict(toks) == gold


def test_unknown_word_uses_capitalization_and_suffix():
    m = HMMTagger(TAGS).fit(TOY)
    toks = ["Zed", "Quinn", "visited", "Berlin", "."]      # all names/places unseen
    pred = m.predict(toks)
    assert len(pred) == 5 and pred[2] == "O" and pred[4] == "O"
    assert pred[0] == "B-PER"                              # initcap + position prior


def test_uniform_fallback_for_unseen_suffix():
    m = HMMTagger(TAGS).fit(TOY)
    e = m._oov_log_emission("\u4e2d\u6587")                # chars never seen in training
    assert np.allclose(np.exp(e), 1 / len(TAGS))


def test_cap_class():
    assert cap_class("EU") == "allcap" and cap_class("Paris") == "initcap"
    assert cap_class("dog") == "lower" and cap_class("1996") == "digit"
