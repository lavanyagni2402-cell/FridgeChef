"""Hyper-parameter tuning on a temporal validation split carved out of TRAIN only
(newest 20% of each user's training interactions). The test set is never touched."""
import itertools
import sys
import time

import numpy as np
import pandas as pd

from common import Data, evaluate_model, save_json, PROC, SEED
from models import UserKNN, ItemKNN, PureSVD, ALS, BPR, ContentKNN, NeuMF


def validation_data():
    tr = pd.read_pickle(PROC / "train.pkl").sort_values(["user_id", "date", "recipe_id"])
    tr["rank"] = tr.groupby("user_id").cumcount()
    tr["n"] = tr.groupby("user_id")["recipe_id"].transform("size")
    nv = np.maximum(1, np.round(tr["n"] * 0.2)).astype(int)
    is_val = tr["rank"] >= tr["n"] - nv
    return Data(train=tr[~is_val].drop(columns=["rank", "n"]), test=tr[is_val].drop(columns=["rank", "n"]))


def main():
    d = validation_data()
    users = np.random.default_rng(SEED).choice(d.test_users, 2500, replace=False)
    grids = {
        "User-kNN": (UserKNN, [dict(k=k) for k in (20, 50, 100, 200, 400)]),
        "Item-kNN": (ItemKNN, [dict(k=k) for k in (20, 50, 100, 200, 500)]),
        "PureSVD": (PureSVD, [dict(f=f) for f in (10, 20, 50, 100, 200)]),
        "ALS": (ALS, [dict(f=f, reg=r, alpha=a) for f, r, a in
                      itertools.product((32, 64, 128), (0.01, 0.1, 1.0), (1, 5, 20, 40))]),
        "BPR": (BPR, [dict(f=f, lr=lr, reg=rg, iters=it) for f, lr, rg, it in
                      itertools.product((32, 64), (0.01, 0.05), (0.001, 0.01), (100, 300))]),
        "Content": (ContentKNN, [dict(svd_dims=s) for s in (None, 100, 300)]),
        "NeuMF": (NeuMF, [dict(epochs=e) for e in (2, 4, 8)]),
    }
    only = sys.argv[1:] or list(grids)
    out = {}
    for name in only:
        cls, grid = grids[name]
        rows = []
        for params in grid:
            t = time.time()
            m = cls(**params).fit(d)
            r, _ = evaluate_model(m, d, users=users)
            rows.append({"params": params, "NDCG@10": r["NDCG@10"], "HR@10": r["HR@10"], "secs": time.time() - t})
            print(name, params, round(r["NDCG@10"], 4), round(r["HR@10"], 4), flush=True)
        best = max(rows, key=lambda x: x["NDCG@10"])
        out[name] = {"best": best, "all": rows}
        print("BEST", name, best, flush=True)
    save_json(out, "tuning_" + "_".join(only) + ".json")


if __name__ == "__main__":
    main()
