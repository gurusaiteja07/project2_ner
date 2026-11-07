"""Training / evaluation loop shared by Task 2 and Task 3."""
import copy
import time

import torch
from seqeval.metrics import f1_score

from nn_data import batches, make_batch


def evaluate(model, sents, vocab, device, bs=64):
    model.eval()
    gold, pred = [], []
    for chunk in batches(sents, bs, shuffle=False):
        b = make_batch(chunk, vocab, device=device)
        for (toks, tgs), p in zip(chunk, model.predict(b)):
            gold.append(tgs)
            pred.append([vocab.tags[i] for i in p])
    return f1_score(gold, pred), gold, pred


def train(model, train_set, val_set, vocab, cfg, device, ckpt_path):
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"])
    best_f1, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(cfg["max_epochs"]):
        lr = cfg["lr"] / (1 + cfg["lr_decay"] * epoch)         # lr_t = lr0 / (1 + 0.05 t)
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        t0, total = time.time(), 0.0
        for chunk in batches(train_set, cfg["batch_size"], shuffle=True):
            b = make_batch(chunk, vocab, train=True, unk_prob=cfg["singleton_unk_prob"], device=device)
            opt.zero_grad()
            loss = model.loss(b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            total += loss.item()
        val_f1, _, _ = evaluate(model, val_set, vocab, device)
        history.append(dict(epoch=epoch + 1, lr=lr, loss=total / len(train_set), val_f1=val_f1))
        print(f"epoch {epoch+1:2d} lr={lr:.5f} loss/sent={total/len(train_set):8.3f} "
              f"val F1={val_f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if val_f1 > best_f1:
            best_f1, bad = val_f1, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, ckpt_path)
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"early stop: no val improvement for {cfg['patience']} epochs")
                break
    model.load_state_dict(best_state)
    return best_f1, history
