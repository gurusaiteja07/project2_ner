import itertools

import pytest

torch = pytest.importorskip("torch")
from crf import CRF


def brute(crf, em, L):
    T = em.size(1)
    scores = {}
    for p in itertools.product(range(T), repeat=L):
        s = crf.start[p[0]] + em[0, p[0]] + crf.end[p[-1]]
        for i in range(1, L):
            s = s + crf.trans[p[i - 1], p[i]] + em[i, p[i]]
        scores[p] = s
    return scores


def test_partition_score_and_decode_match_brute_force():
    torch.manual_seed(0)
    T, L = 3, 4
    crf = CRF(T)
    em = torch.randn(2, L, T)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)   # 2nd sentence has length 3
    tags = torch.tensor([[0, 1, 2, 1], [2, 2, 0, 0]])
    nll = crf.nll(em, tags, mask)
    for b, n in enumerate([4, 3]):
        sc = brute(crf, em[b, :n], n)
        logz = torch.logsumexp(torch.stack(list(sc.values())), 0)
        gold = sc[tuple(tags[b, :n].tolist())]
        assert torch.allclose(nll[b], logz - gold, atol=1e-5)
        assert nll[b] >= 0
        assert crf.decode(em, mask)[b] == list(max(sc, key=lambda k: sc[k].item()))
