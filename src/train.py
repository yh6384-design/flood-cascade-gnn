"""
Train the GNN and the MLP baseline on the simulated scenario set, and score
them the way the problem actually needs to be scored.

Two things matter here and both are easy to get wrong.

1. Split by EVENT, not by node. Every node in an event shares the same flood,
   the same graph and the same cascade. If nodes from one event land in both
   train and test, the model can memorise that event and the metrics become
   meaningless.

2. Report the SECONDARY-failure metric separately. A substation that the flood
   destroyed directly is trivially predictable: feature 2 says so. The
   interesting question is whether a model can predict the substations that
   survived the water but went down anyway because the network around them
   collapsed. Overall AUC hides this. The secondary metric is the honest one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from models import build

IDX_INITIAL = 2  # column of `initially_failed` in the feature matrix


def load_dataset(path: Path, seed: int = 0, split=(0.7, 0.15, 0.15)):
    z = np.load(path, allow_pickle=True)
    X, Y, ei = z["X"], z["Y"], z["edge_index"]
    n_events = X.shape[0]

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_events)
    n_tr = int(split[0] * n_events)
    n_va = int(split[1] * n_events)
    idx = {
        "train": order[:n_tr],
        "val": order[n_tr:n_tr + n_va],
        "test": order[n_tr + n_va:],
    }

    # Standardise features using TRAIN events only.
    tr = X[idx["train"]].reshape(-1, X.shape[-1])
    mu, sd = tr.mean(0), tr.std(0)
    sd[sd < 1e-6] = 1.0

    edge_index = torch.as_tensor(ei, dtype=torch.long)

    def pack(keys):
        out = []
        for e in keys:
            xn = (X[e] - mu) / sd
            out.append(
                Data(
                    x=torch.as_tensor(xn, dtype=torch.float32),
                    edge_index=edge_index,
                    y=torch.as_tensor(Y[e], dtype=torch.float32),
                    initial=torch.as_tensor(X[e, :, IDX_INITIAL] > 0.5),
                )
            )
        return out

    return {k: pack(v) for k, v in idx.items()}, z, (mu, sd)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    p, y, init = [], [], []
    for batch in loader:
        batch = batch.to(device)
        logit = model(batch.x, batch.edge_index)
        p.append(torch.sigmoid(logit).cpu().numpy())
        y.append(batch.y.cpu().numpy())
        init.append(batch.initial.cpu().numpy())
    return np.concatenate(p), np.concatenate(y), np.concatenate(init)


def score(p, y, init) -> dict:
    """Overall metrics, plus metrics restricted to nodes the flood did not hit."""
    out = {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "positive_rate": float(y.mean()),
    }
    sec = ~init  # nodes that survived the direct hazard
    if sec.sum() and 0 < y[sec].sum() < sec.sum():
        out["secondary_roc_auc"] = float(roc_auc_score(y[sec], p[sec]))
        out["secondary_pr_auc"] = float(average_precision_score(y[sec], p[sec]))
        out["secondary_positive_rate"] = float(y[sec].mean())
    return out


def train_one(name, data, in_dim, args, device):
    model = build(name, in_dim, hidden=args.hidden, layers=args.layers,
                  dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    ytr = np.concatenate([d.y.numpy() for d in data["train"]])
    pos_weight = torch.tensor([(1 - ytr.mean()) / max(ytr.mean(), 1e-6)], device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    tl = DataLoader(data["train"], batch_size=args.batch_size, shuffle=True)
    vl = DataLoader(data["val"], batch_size=args.batch_size)
    sl = DataLoader(data["test"], batch_size=args.batch_size)

    best, best_state, patience = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in tl:
            batch = batch.to(device)
            opt.zero_grad()
            loss = lossf(model(batch.x, batch.edge_index), batch.y)
            loss.backward()
            opt.step()
            total += float(loss) * batch.num_graphs
        total /= len(data["train"])

        vp, vy, vi = predict(model, vl, device)
        v = score(vp, vy, vi)
        key = v.get("secondary_pr_auc", v["pr_auc"])
        if key > best:
            best, patience = key, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            patience += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{name}] epoch {epoch:3d}  loss {total:.4f}  "
                  f"val PR {v['pr_auc']:.3f}  val secondary PR {key:.3f}")
        if patience >= args.patience:
            print(f"  [{name}] early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    tp, ty, ti = predict(model, sl, device)
    return model, score(tp, ty, ti), (tp, ty, ti)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the cascade surrogate models.")
    ap.add_argument("--data", type=Path, default=Path("data/scenarios.npz"))
    ap.add_argument("--outdir", type=Path, default=Path("artifacts"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, _, _ = load_dataset(args.data, seed=args.seed)
    in_dim = data["train"][0].x.shape[1]
    print(f"events: train={len(data['train'])} val={len(data['val'])} "
          f"test={len(data['test'])}  features={in_dim}  device={device}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    results, preds = {}, {}
    for name in ("gnn", "mlp"):
        print(f"\ntraining {name}")
        model, s, pr = train_one(name, data, in_dim, args, device)
        results[name] = s
        preds[name] = pr
        torch.save(model.state_dict(), args.outdir / f"{name}.pt")

    np.savez_compressed(
        args.outdir / "test_predictions.npz",
        gnn=preds["gnn"][0], mlp=preds["mlp"][0],
        y=preds["gnn"][1], initial=preds["gnn"][2],
    )
    (args.outdir / "metrics.json").write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 62)
    print(f"{'metric':28s} {'GNN':>10s} {'MLP':>10s} {'delta':>10s}")
    print("-" * 62)
    for k in results["gnn"]:
        g, m = results["gnn"][k], results["mlp"][k]
        print(f"{k:28s} {g:10.4f} {m:10.4f} {g - m:+10.4f}")
    print("=" * 62)
    print(f"\nwrote {args.outdir}/metrics.json")


if __name__ == "__main__":
    main()
