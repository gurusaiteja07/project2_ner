"""Char+word BiLSTM tagger. use_crf=True -> Lample et al. BiLSTM-CRF (Task 2).
use_crf=False -> same network with a softmax output (used by the Task 3 hybrid)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from crf import CRF


class BiLSTMTagger(nn.Module):
    def __init__(self, n_chars, n_tags, pretrained, char_dim=25, char_hidden=25,
                 hidden=100, dropout=0.5, use_crf=True):
        super().__init__()
        word_dim = pretrained.size(1)
        self.char_emb = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.char_lstm = nn.LSTM(char_dim, char_hidden, batch_first=True, bidirectional=True)
        self.word_emb = nn.Embedding.from_pretrained(pretrained.clone(), freeze=False, padding_idx=0)  # fine-tuned
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(word_dim + 2 * char_hidden, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, n_tags)
        self.use_crf = use_crf
        self.crf = CRF(n_tags) if use_crf else None
        self.char_out = 2 * char_hidden

    def char_features(self, chars, clen, mask):
        B, L, C = chars.shape
        flat = mask.reshape(-1)
        c = chars.reshape(B * L, C)[flat]
        n = clen.reshape(-1)[flat]
        packed = pack_padded_sequence(self.char_emb(c), n.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.char_lstm(packed)                       # h: (2, N, H)
        v = torch.cat([h[0], h[1]], dim=-1)                      # final fwd + bwd states
        out = chars.new_zeros(B * L, self.char_out, dtype=v.dtype)
        out[flat] = v
        return out.view(B, L, -1)

    def emissions(self, batch):
        x = torch.cat([self.word_emb(batch["words"]),
                       self.char_features(batch["chars"], batch["clen"], batch["mask"])], dim=-1)
        x = self.drop(x)
        L = x.size(1)
        packed = pack_padded_sequence(x, batch["lengths"].cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=L)
        return self.out(self.drop(h))                            # (B, L, T)

    def loss(self, batch):
        """Summed over the sentences in the batch (matches the paper's per-sentence updates)."""
        em = self.emissions(batch)
        if self.use_crf:
            return self.crf.nll(em, batch["tags"], batch["mask"]).sum()
        # token-level softmax cross-entropy, summed over tokens
        return F.cross_entropy(em[batch["mask"]], batch["tags"][batch["mask"]], reduction="sum")

    @torch.no_grad()
    def predict(self, batch):
        em = self.emissions(batch)
        if self.use_crf:
            return self.crf.decode(em, batch["mask"])
        pred = em.argmax(-1)
        return [pred[b, :int(n)].tolist() for b, n in enumerate(batch["lengths"])]

    @torch.no_grad()
    def log_probs(self, batch):
        """Per-token log P(tag | sentence) as list of (n_i, T) numpy arrays (for the hybrid)."""
        lp = F.log_softmax(self.emissions(batch), dim=-1).cpu().numpy()
        return [lp[b, :int(n)] for b, n in enumerate(batch["lengths"])]
