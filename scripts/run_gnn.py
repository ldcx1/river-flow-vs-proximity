"""GAT vs MLP on monthly graph snapshots (paper: GAT vs MLP).

A GAT passes messages along upstream-to-downstream edges. The control is the
same model with self-loops only (an MLP). Features and segment-grouped CV match
the XGBoost panel, and standardisation is fit on training nodes only. Repeats
over --seeds seeds (default 5) with fixed folds. Needs the ``gnn`` extra.
Progress lines go to stderr.

    python scripts/run_gnn.py > results/national/gnn.txt      # ~25 min
"""
import argparse
import sys
import time

import numpy as np

import _bootstrap
from experiments.cv import grouped_folds  # noqa: E402
from experiments.metrics import average_precision, roc_auc  # noqa: E402

SEED, FOLDS, EPOCHS = 0, 5, 60
torch = nn = F = GATConv = add_self_loops = GAT = None


def _load_torch():
    """Import the GNN stack (the ``gnn`` extra) only when the experiment runs."""
    global torch, nn, F, GATConv, add_self_loops, GAT
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv
    from torch_geometric.utils import add_self_loops
    torch.manual_seed(SEED)

    class GAT(nn.Module):
        def __init__(self, in_dim, hid=32, heads=4, p=0.3):
            super().__init__()
            self.g1 = GATConv(in_dim, hid, heads=heads, dropout=p)
            self.g2 = GATConv(hid * heads, hid, heads=1, dropout=p)
            self.lin = nn.Linear(hid, 1)
            self.p = p

        def forward(self, x, ei):
            x = F.elu(self.g1(x, ei))
            x = F.dropout(x, p=self.p, training=self.training)
            x = F.elu(self.g2(x, ei))
            return self.lin(x).squeeze(-1)



def oof(Xraw, ei, labels, fold_of, use_edges, seed=SEED):
    """Out-of-fold node probabilities.

    ``seed`` varies weight init and epoch shuffle. The folds (fold_of) stay fixed."""
    torch.manual_seed(seed)
    M, N, Fd = Xraw.shape
    self_ei = add_self_loops(ei, num_nodes=N)[0]
    edges = self_ei if use_edges else add_self_loops(
        torch.empty((2, 0), dtype=torch.long), num_nodes=N)[0]
    nodes = labels[:, 1]
    out_p, out_y = [], []
    rng = np.random.default_rng(seed)
    for k in range(FOLDS):
        test_nodes = set(np.where(fold_of == k)[0].tolist())
        keep = np.array([n not in test_nodes for n in range(N)])
        flat_tr = Xraw[:, keep, :].reshape(-1, Fd)
        mu = np.nanmean(flat_tr, axis=0); mu = np.where(np.isfinite(mu), mu, 0.0)
        sd = np.nanstd(flat_tr, axis=0); sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        X = torch.tensor(np.nan_to_num((Xraw - mu) / sd, nan=0.0).astype(np.float32))
        tr = labels[np.array([n not in test_nodes for n in nodes])]
        te = labels[np.array([n in test_nodes for n in nodes])]
        tr_by_m = {}
        for mi, ni, y in tr:
            tr_by_m.setdefault(int(mi), []).append((int(ni), float(y)))
        pos = (tr[:, 2] == 1).sum(); neg = (tr[:, 2] == 0).sum()
        pw = torch.tensor([neg / max(1, pos)], dtype=torch.float32)
        model = GAT(Fd)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        model.train()
        ms = list(tr_by_m.keys())
        for ep in range(EPOCHS):
            rng.shuffle(ms)
            for mi in ms:
                cells = tr_by_m[mi]
                ni = torch.tensor([c[0] for c in cells])
                yy = torch.tensor([c[1] for c in cells])
                opt.zero_grad()
                logit = model(X[mi], edges)
                loss = lossf(logit[ni], yy)
                loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            for mi in sorted(set(int(m) for m in te[:, 0])):
                logit = torch.sigmoid(model(X[mi], edges))
                for r in te[te[:, 0] == mi]:
                    out_p.append(float(logit[int(r[1])])); out_y.append(int(r[2]))
    return np.array(out_y), np.array(out_p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshots", default=_bootstrap.data_de("graph_snapshots_national.npz"),
                    help="monthly graph snapshots (.npz with X, edge_index, labels)")
    ap.add_argument("--seeds", type=int, default=5,
                    help="number of training seeds per model (folds stay fixed)")
    args = ap.parse_args()
    _load_torch()
    t0 = time.time()

    d = np.load(args.snapshots, allow_pickle=True)
    X = d["X"].astype(np.float32)
    ei = torch.tensor(d["edge_index"], dtype=torch.long)
    labels = d["labels"].astype(np.int64)
    M, N, Fd = X.shape
    print(f"snapshots: {M} months x {N} nodes x {Fd} feats | edges {ei.shape[1]} | "
          f"labeled cells {len(labels)} (pos {int(labels[:,2].sum())})")

    row_fold = grouped_folds(labels[:, 1], labels[:, 2], FOLDS, SEED)
    fold_of = np.full(N, -1)
    for (mi, ni, yy), f in zip(labels, row_fold):
        fold_of[int(ni)] = int(f)

    prev = labels[:, 2].mean()
    n_seeds = args.seeds
    print(f"prevalence {prev:.3f} | {n_seeds} seeds (segment-grouped folds fixed; "
          f"weight init + epoch shuffle varied)\n")
    print(f"{'model':<26} {'ROC-AUC (mean+/-sd)':>22} {'PR-AUC (mean+/-sd)':>22}")
    print("-" * 72)
    summary = {}
    for name, use in [("MLP (no edges)", False), ("GAT (topology, directed)", True)]:
        rocs, prs = [], []
        for s in range(n_seeds):
            # Progress goes to stderr so a redirected log holds only the results.
            print(f"[{time.time() - t0:6.0f}s] {name}: seed {s + 1}/{n_seeds}",
                  file=sys.stderr, flush=True)
            y, p = oof(X, ei, labels, fold_of, use, seed=s)
            rocs.append(roc_auc(y, p)); prs.append(average_precision(y, p))
        rocs, prs = np.array(rocs), np.array(prs)
        summary[name] = (rocs, prs)
        print(f"{name:<26} {rocs.mean():>10.3f} +/- {rocs.std():.3f}   "
              f"{prs.mean():>10.3f} +/- {prs.std():.3f}")
    gat_roc = summary["GAT (topology, directed)"][0]
    mlp_roc = summary["MLP (no edges)"][0]
    print(f"\nper-seed ROC-AUC  MLP: {np.round(mlp_roc,3).tolist()}")
    print(f"per-seed ROC-AUC  GAT: {np.round(gat_roc,3).tolist()}")
    delta = (gat_roc - mlp_roc)
    verdict = ("the graph layer does not beat the no-edge control"
               if delta.mean() <= 0 else
               "the graph layer is nominally ahead of the no-edge control")
    print(f"\nGAT - MLP ROC delta: {delta.mean():+.3f} +/- {delta.std():.3f} "
          f"over {n_seeds} seeds ({verdict}).")
    print("CV = positive-rate-balanced segment-grouped 5-fold (no month leakage); tie-aware AUC/AP.")


if __name__ == "__main__":
    main()
