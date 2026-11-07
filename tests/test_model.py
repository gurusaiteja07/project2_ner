import pytest

torch = pytest.importorskip("torch")
from model import BiLSTMTagger
from nn_data import Vocab, make_batch

TAGS = ["O", "B-PER", "I-PER", "B-LOC"]
DATA = [(["John", "Smith", "visited", "Paris", "."], ["B-PER", "I-PER", "O", "B-LOC", "O"]),
        (["Hi", "Mary", "."], ["O", "B-PER", "O"])]


def build(use_crf):
    v = Vocab(DATA, [], TAGS, glove_path=None, dim=100)
    return v, BiLSTMTagger(len(v.c2i), len(TAGS), v.emb, use_crf=use_crf)


@pytest.mark.parametrize("use_crf", [True, False])
def test_shapes_and_overfit(use_crf):
    torch.manual_seed(0)
    v, m = build(use_crf)
    b = make_batch(DATA, v)
    assert m.emissions(b).shape == (2, 5, len(TAGS))
    opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
    first = m.loss(b).item()
    for _ in range(150):
        opt.zero_grad(); m.loss(b).backward(); opt.step()
    m.eval()
    assert m.loss(b).item() < first * 0.2                       # loss drops
    pred = m.predict(b)
    assert [[v.tags[i] for i in p] for p in pred] == [t for _, t in DATA]   # memorises 2 sentences


def test_padding_does_not_change_output():
    torch.manual_seed(0)
    v, m = build(True)
    m.eval()
    alone = m.emissions(make_batch([DATA[1]], v))[0]
    padded = m.emissions(make_batch(DATA, v))[1, :3]
    assert torch.allclose(alone, padded, atol=1e-5)
