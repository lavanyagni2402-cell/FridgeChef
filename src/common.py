"""Shared data loading, matrices and evaluation metrics for FridgeChef."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
SEED = 42
K = 10


class Data:
    """Index maps + sparse train matrices + test ground truth."""

    def __init__(self, train=None, test=None):
        self.recipes = pd.read_pickle(PROC / "recipes_all.pkl").set_index("recipe_id")
        self.train = pd.read_pickle(PROC / "train.pkl") if train is None else train
        self.test = pd.read_pickle(PROC / "test.pkl") if test is None else test
        users = np.sort(self.train.user_id.unique())
        items = np.sort(self.train.recipe_id.unique())
        self.uid = {u: i for i, u in enumerate(users)}
        self.iid = {r: i for i, r in enumerate(items)}
        self.users, self.items = users, items
        self.n_users, self.n_items = len(users), len(items)
        u = self.train.user_id.map(self.uid).values
        i = self.train.recipe_id.map(self.iid).values
        # implicit: every interaction (rated or not) = "cooked it"
        self.X = sp.csr_matrix((np.ones(len(u), np.float32), (u, i)), shape=(self.n_users, self.n_items))
        r = self.train.rating.values.astype(np.float32)
        m = r > 0
        self.R = sp.csr_matrix((r[m], (u[m], i[m])), shape=(self.n_users, self.n_items))
        self.pop = np.asarray(self.X.sum(0)).ravel()
        t = self.test[self.test.recipe_id.isin(self.iid) & self.test.user_id.isin(self.uid)]
        self.truth = {self.uid[k]: set(v.map(self.iid)) for k, v in t.groupby("user_id").recipe_id}
        self.test_users = np.array(sorted(self.truth))
        self.item_meta = self.recipes.loc[self.items]


# ----------------------------------------------------------------- metrics
def ranking_metrics(topk, truth_sets, n_items, pop, n_users, k=K):
    """topk: (n, k) item indices; truth_sets: list of sets."""
    P, R, F, H, N, AP = [], [], [], [], [], []
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    for rec, tru in zip(topk, truth_sets):
        hits = np.array([r in tru for r in rec[:k]], dtype=float)
        nh = hits.sum()
        p = nh / k
        r = nh / len(tru)
        P.append(p); R.append(r); H.append(float(nh > 0))
        F.append(0 if nh == 0 else 2 * p * r / (p + r))
        idcg = disc[:min(len(tru), k)].sum()
        N.append((hits * disc).sum() / idcg)
        if nh:
            prec_at = np.cumsum(hits) / np.arange(1, k + 1)
            AP.append((prec_at * hits).sum() / min(len(tru), k))
        else:
            AP.append(0.0)
    flat = np.asarray(topk)[:, :k].ravel()
    novelty = float(np.mean(-np.log2(np.maximum(pop[flat], 1) / n_users)))
    return {"P@10": np.mean(P), "R@10": np.mean(R), "F1@10": np.mean(F), "HR@10": np.mean(H),
            "NDCG@10": np.mean(N), "MAP@10": np.mean(AP),
            "Coverage": len(np.unique(flat)) / n_items, "Novelty": novelty}


def topk_from_scores(scores, seen_csr_rows, k=K):
    """Mask seen items, return top-k indices per row."""
    s = np.array(scores, dtype=np.float32, copy=True)
    for row in range(s.shape[0]):
        a, b = seen_csr_rows.indptr[row], seen_csr_rows.indptr[row + 1]
        s[row, seen_csr_rows.indices[a:b]] = -np.inf
    idx = np.argpartition(-s, k, axis=1)[:, :k]
    order = np.take_along_axis(s, idx, 1).argsort(1)[:, ::-1]
    return np.take_along_axis(idx, order, 1)


def evaluate_model(model, data, users=None, batch=1000, k=K, seen=None):
    users = data.test_users if users is None else users
    seen = data.X if seen is None else seen
    tops = []
    for s in range(0, len(users), batch):
        ub = users[s:s + batch]
        tops.append(topk_from_scores(model.score(ub), seen[ub], k))
    top = np.vstack(tops)
    return ranking_metrics(top, [data.truth[u] for u in users], data.n_items, data.pop, data.n_users, k), top


def save_json(obj, name):
    (RES / name).write_text(json.dumps(obj, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o)))
