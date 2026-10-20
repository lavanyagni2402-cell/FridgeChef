"""Experiment 1 - explicit rating prediction (error metrics), normalisation, missing values.
Uses the Surprise library on the explicit 1-5 star ratings of the 10-core data."""
import time

import numpy as np
import pandas as pd
from surprise import (Dataset, Reader, NormalPredictor, BaselineOnly, KNNBasic, KNNWithMeans,
                      KNNWithZScore, SVD, SVDpp, NMF, accuracy)

from common import PROC, save_json, SEED

tr = pd.read_pickle(PROC / "train.pkl")
te = pd.read_pickle(PROC / "test.pkl")
te = te[te.user_id.isin(tr.user_id) & te.recipe_id.isin(tr.recipe_id)]
reader = Reader(rating_scale=(1, 5))


def build(train_df):
    ds = Dataset.load_from_df(train_df[["user_id", "recipe_id", "rating"]], reader)
    return ds.build_full_trainset()


tr_exp = tr[tr.rating > 0]
te_exp = te[te.rating > 0]
testset = list(te_exp[["user_id", "recipe_id", "rating"]].itertuples(index=False, name=None))
trainset = build(tr_exp)

results = {"n_train_ratings": len(tr_exp), "n_test_ratings": len(testset),
           "test_rating_dist": te_exp.rating.value_counts(normalize=True).sort_index().round(4).to_dict()}

# constant baselines
truth = te_exp.rating.values
gm = tr_exp.rating.mean()
results["Global mean"] = {"RMSE": float(np.sqrt(np.mean((truth - gm) ** 2))), "MAE": float(np.mean(np.abs(truth - gm)))}
results["Always 5 stars"] = {"RMSE": float(np.sqrt(np.mean((truth - 5) ** 2))), "MAE": float(np.mean(np.abs(truth - 5)))}

user_sim = lambda name: {"name": name, "user_based": True, "min_support": 2}
item_sim = lambda name: {"name": name, "user_based": False, "min_support": 2}
algos = {
    "Random (rating distribution)": NormalPredictor(),
    "Baseline (user + item bias)": BaselineOnly(bsl_options={"method": "als"}),
    "User-kNN raw (cosine)": KNNBasic(k=40, sim_options=user_sim("cosine"), verbose=False),
    "User-kNN mean-centred (Pearson)": KNNWithMeans(k=40, sim_options=user_sim("pearson"), verbose=False),
    "User-kNN z-score (Pearson)": KNNWithZScore(k=40, sim_options=user_sim("pearson"), verbose=False),
    "SVD (biased MF, SGD)": SVD(n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.05, random_state=SEED),
    "SVD++": SVDpp(n_factors=20, n_epochs=15, random_state=SEED),
    "NMF": NMF(n_factors=15, n_epochs=50, random_state=SEED),
}
for name, algo in algos.items():
    t = time.time()
    algo.fit(trainset)
    preds = algo.test(testset)
    results[name] = {"RMSE": accuracy.rmse(preds, verbose=False), "MAE": accuracy.mae(preds, verbose=False),
                     "secs": round(time.time() - t, 1)}
    print(name, results[name], flush=True)

# ---- item-based kNN, own NumPy implementation (Surprise needs ~4 dense float64 n_items^2 arrays)
import scipy.sparse as sp
from sklearn.preprocessing import normalize as l2


def item_knn_predict(variant, k=40):
    users = np.sort(tr_exp.user_id.unique()); items = np.sort(tr_exp.recipe_id.unique())
    ui = {u: j for j, u in enumerate(users)}; ii = {r: j for j, r in enumerate(items)}
    rows = tr_exp.user_id.map(ui).values; cols = tr_exp.recipe_id.map(ii).values
    r = tr_exp.rating.values.astype(np.float32)
    R = sp.csr_matrix((r, (rows, cols)), shape=(len(users), len(items)))
    cnt = np.asarray((R > 0).sum(0)).ravel()
    mu = np.asarray(R.sum(0)).ravel() / np.maximum(cnt, 1)
    sq = np.asarray(R.multiply(R).sum(0)).ravel() / np.maximum(cnt, 1)
    sd = np.sqrt(np.maximum(sq - mu ** 2, 1e-6))
    sd = np.where(sd < 0.25, 0.25, sd)                  # floor: many recipes have only 5-star ratings
    V = R.copy().tocsc()
    if variant != "raw":
        V.data = V.data - np.repeat(mu, np.diff(V.indptr))   # csc: column j's entries are contiguous
    if variant == "zscore":
        V.data = V.data / np.repeat(sd, np.diff(V.indptr))
    Vn = l2(V.T.tocsr())                                # items x users, rows L2-normalised
    S = (Vn @ Vn.T).toarray().astype(np.float32)
    np.fill_diagonal(S, 0)
    Rcsr = R.tocsr()
    preds, truth_v = [], []
    tdf = te_exp[te_exp.recipe_id.isin(ii) & te_exp.user_id.isin(ui)]
    for u, g in tdf.groupby("user_id"):
        a, b = Rcsr.indptr[ui[u]], Rcsr.indptr[ui[u] + 1]
        J, rv = Rcsr.indices[a:b], Rcsr.data[a:b]
        I = g.recipe_id.map(ii).values
        Ssub = S[np.ix_(I, J)]
        kk = min(k, len(J))
        idx = np.argpartition(-Ssub, kk - 1, axis=1)[:, :kk]
        s = np.take_along_axis(Ssub, idx, 1)
        s = np.where(s > 0, s, 0)
        if variant == "raw":
            dev = rv[idx]
            base = np.zeros(len(I))
            scale = np.ones(len(I))
        elif variant == "mean":
            dev = rv[idx] - mu[J][idx]; base = mu[I]; scale = np.ones(len(I))
        else:
            dev = (rv[idx] - mu[J][idx]) / sd[J][idx]; base = mu[I]; scale = sd[I]
        den = s.sum(1)
        est = np.where(den > 0, base + scale * (s * dev).sum(1) / np.maximum(den, 1e-9), mu[I] if variant != "raw" else gm)
        preds.append(np.clip(est, 1, 5)); truth_v.append(g.rating.values)
    p, t = np.concatenate(preds), np.concatenate(truth_v)
    return {"RMSE": float(np.sqrt(np.mean((p - t) ** 2))), "MAE": float(np.mean(np.abs(p - t)))}


for name, var in [("Item-kNN raw (cosine)", "raw"), ("Item-kNN mean-centred (Pearson)", "mean"),
                  ("Item-kNN z-score (Pearson)", "zscore")]:
    t0 = time.time()
    results[name] = {**item_knn_predict(var), "secs": round(time.time() - t0, 1)}
    print(name, results[name], flush=True)

# ---- missing values: reviews with rating 0 (no stars given)
mv = {}
for strategy in ["drop (treat as missing)", "impute user mean", "impute item mean", "treat as 1 star (naive)"]:
    d = tr.copy()
    d["rating"] = d["rating"].astype(float)
    miss = d.rating == 0
    if strategy.startswith("drop"):
        d = d[~miss]
    elif strategy == "impute user mean":
        um = tr_exp.groupby("user_id").rating.mean()
        d.loc[miss, "rating"] = d.loc[miss, "user_id"].map(um).fillna(gm)
    elif strategy == "impute item mean":
        im = tr_exp.groupby("recipe_id").rating.mean()
        d.loc[miss, "rating"] = d.loc[miss, "recipe_id"].map(im).fillna(gm)
    else:
        d.loc[miss, "rating"] = 1  # lowest valid star = what "0 means bad" would imply
    algo = SVD(n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.05, random_state=SEED)
    algo.fit(build(d))
    p = algo.test(testset)
    mv[strategy] = {"RMSE": accuracy.rmse(p, verbose=False), "MAE": accuracy.mae(p, verbose=False)}
    print(strategy, mv[strategy], flush=True)
results["missing_value_strategies"] = mv
results["n_missing_in_train"] = int((tr.rating == 0).sum())
save_json(results, "exp_rating.json")
