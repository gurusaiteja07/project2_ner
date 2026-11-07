"""Batched linear-chain CRF (own implementation, no external package)."""
import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, n_tags):
        super().__init__()
        self.n = n_tags
        self.trans = nn.Parameter(torch.zeros(n_tags, n_tags))   # trans[i, j]: i -> j
        self.start = nn.Parameter(torch.zeros(n_tags))
        self.end = nn.Parameter(torch.zeros(n_tags))
        for p in (self.trans, self.start, self.end):
            nn.init.uniform_(p, -0.1, 0.1)

    def score(self, em, tags, mask):
        """Score of the gold path. em (B,L,T), tags (B,L), mask (B,L) bool."""
        B, L, _ = em.shape
        m = mask.float()
        s = self.start[tags[:, 0]] + em[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for i in range(1, L):
            step = self.trans[tags[:, i - 1], tags[:, i]] + em[:, i].gather(1, tags[:, i:i + 1]).squeeze(1)
            s = s + step * m[:, i]
        last = tags.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
        return s + self.end[last]

    def log_partition(self, em, mask):
        B, L, _ = em.shape
        alpha = self.start + em[:, 0]                              # (B,T)
        for i in range(1, L):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.trans.unsqueeze(0) + em[:, i].unsqueeze(1), dim=1)
            alpha = torch.where(mask[:, i].unsqueeze(1), nxt, alpha)
        return torch.logsumexp(alpha + self.end, dim=1)

    def nll(self, em, tags, mask):
        """Per-sentence negative log-likelihood, shape (B,)."""
        return self.log_partition(em, mask) - self.score(em, tags, mask)

    @torch.no_grad()
    def decode(self, em, mask):
        B, L, _ = em.shape
        score = self.start + em[:, 0]
        hist = []
        for i in range(1, L):
            best, idx = (score.unsqueeze(2) + self.trans.unsqueeze(0)).max(1)
            score = torch.where(mask[:, i].unsqueeze(1), best + em[:, i], score)
            hist.append(idx)
        score = score + self.end
        last = score.argmax(1)
        lengths = mask.sum(1).tolist()
        paths = []
        for b in range(B):
            tag = int(last[b])
            path = [tag]
            for step in range(lengths[b] - 1, 0, -1):
                tag = int(hist[step - 1][b, tag])
                path.append(tag)
            paths.append(path[::-1])
        return paths
