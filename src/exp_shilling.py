"""Experiment 5 - shilling (push) attacks and a detection-based defence.

Attacker goal: push an unpopular target recipe into genuine users' top-10.
Fake profile = target (5 stars) + 20 filler recipes.
  Random     fillers drawn uniformly, ratings ~ global rating distribution
  Average    fillers drawn uniformly, rating = filler's own mean rating
  Bandwagon  fillers = most popular recipes, rated 5
  Segment    fillers = recipes most similar in content to the target, rated 5
Attack size = number of fake profiles as % of genuine users (1%, 3%, 5%).
Measured on Item-kNN and ALS: Hit ratio of the target = share of genuine users who get it in their top-10.
Explicit SVD: prediction shift = mean change of the target's predicted rating.
Defence: a Random-Forest detector over profile features (Chirita 2005; Williams 2007)."""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import normalize
from surprise import Dataset, Reader, SVD

from common import Data, PROC, RES, SEED, save_json
from models import ItemKNN, ALS, ContentKNN

rng = np.random.default_rng(SEED)
FILLER = 20
SIZES = (0.01, 0.03, 0.05)
ATTACKS = ("Random", "Average", "Bandwagon", "Segment")

tr = pd.read_pickle(PROC / "train.pkl")
te = pd.read_pickle(PROC / "test.pkl")
base = Data()
als_p = json.loads((RES / "tuning_ALS_BPR.json").read_text())["ALS"]["best"]["params"]
iknn_p = json.loads((RES / "tuning_User-kNN_Item-kNN_PureSVD_Content_NeuMF.json").read_text())["Item-kNN"]["best"]["params"]
content = ContentKNN().fit(base)
exp = tr[tr.rating > 0]
item_mean = exp.groupby("recipe_id").rating.mean()
rating_dist = exp.rating.value_counts(normalize=True).sort_index()
pop_ids = base.items[np.argsort(-base.pop)[:200]]
n_genuine = base.n_users

pop_series = pd.Series(base.pop, index=base.items)
candidates = pop_series[(pop_series >= 10) & (pop_series <= 12)].index.values
targets = rng.choice(candidates, 4, replace=False)
start = tr.date.max()


def make_profiles(kind, target, n_profiles, uid_start):
    rows = []
    t_idx = base.iid[target]
    sim = np.asarray(content.F @ content.F[t_idx].T.toarray()).ravel() if hasattr(content.F, "toarray") else content.F @ content.F[t_idx]
    seg = base.items[np.argsort(-np.asarray(sim).ravel())[1:61]]
    for p in range(n_profiles):
        uid = uid_start + p
        if kind in ("Random", "Average"):
            fill = rng.choice(base.items, FILLER, replace=False)
        elif kind == "Bandwagon":
            fill = rng.choice(pop_ids, FILLER, replace=False)
        else:
            fill = rng.choice(seg, FILLER, replace=False)
        if kind == "Random":
            r = rng.choice(rating_dist.index.values, FILLER, p=rating_dist.values)
        elif kind == "Average":
            r = np.clip(np.round(item_mean.reindex(fill).fillna(4.4).values), 1, 5)
        else:
            r = np.full(FILLER, 5)
        # fake accounts post their reviews within a 10-day burst
        dates = start + pd.to_timedelta(rng.integers(0, 10 * 24, FILLER + 1), unit="h")
        for it, rt, dt in zip(list(fill) + [target], list(r) + [5], dates):
            rows.append((uid, it, dt, int(rt)))
    return pd.DataFrame(rows, columns=["user_id", "recipe_id", "date", "rating"])


def hit_ratio(model, d, target):
    users = np.array([d.uid[u] for u in base.users])         # genuine users only
    j = d.iid[target]
    hits = 0
    for s in range(0, len(users), 1500):
        ub = users[s:s + 1500]
        S = model.score(ub)
        X = d.X[ub].toarray() > 0
        S[X] = -np.inf
        kth = -np.partition(-S, 9, axis=1)[:, 9]          # 10th highest score per user
        hits += int(((S[:, j] >= kth) & ~X[:, j]).sum())
    return hits / len(users)


def svd_shift(train_df, target, genuine_users):
    ds = Dataset.load_from_df(train_df[train_df.rating > 0][["user_id", "recipe_id", "rating"]], Reader(rating_scale=(1, 5)))
    a = SVD(n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.05, random_state=SEED).fit(ds.build_full_trainset())
    return float(np.mean([a.predict(u, target).est for u in genuine_users]))


out = {"targets": [int(t) for t in targets], "target_pop": [int(pop_series[t]) for t in targets],
       "filler": FILLER, "sizes": SIZES, "n_genuine": n_genuine}
sample_users = rng.choice(base.users, 800, replace=False)

# --- clean baseline
clean = {"Item-kNN": [], "ALS": [], "SVD_pred": []}
for t in targets:
    clean["Item-kNN"].append(hit_ratio(ItemKNN(**iknn_p).fit(base), base, t))
    clean["ALS"].append(hit_ratio(ALS(**als_p).fit(base), base, t))
for t in targets[:1]:
    clean["SVD_pred"].append(svd_shift(tr, t, sample_users))
out["clean"] = {k: float(np.mean(v)) for k, v in clean.items()}
print("clean", out["clean"], flush=True)

rows, injected_all = [], []
uid0 = int(tr.user_id.max()) + 1
for kind in ATTACKS:
    for size in SIZES:
        n_fake = int(round(size * n_genuine))
        hr = {"Item-kNN": [], "ALS": []}
        shifts = []
        for ti, t in enumerate(targets):
            fake = make_profiles(kind, t, n_fake, uid0)
            uid0 += n_fake
            if size == SIZES[-1]:
                injected_all.append(fake.assign(kind=kind, target=t))
            d = Data(train=pd.concat([tr, fake]), test=te)
            hr["Item-kNN"].append(hit_ratio(ItemKNN(**iknn_p).fit(d), d, t))
            hr["ALS"].append(hit_ratio(ALS(**als_p).fit(d), d, t))
            if ti < 1:
                shifts.append(svd_shift(pd.concat([tr, fake]), t, sample_users) - clean["SVD_pred"][ti])
        row = {"attack": kind, "size": size, "n_fake": n_fake,
               "HR_ItemKNN": float(np.mean(hr["Item-kNN"])), "HR_ALS": float(np.mean(hr["ALS"])),
               "SVD_prediction_shift": float(np.mean(shifts))}
        rows.append(row)
        print(row, flush=True)
out["attacks"] = rows

# ------------------------------------------------------------------ detection
fake_all = pd.concat(injected_all)


def features(df, X_users):
    """Generic profile attributes used in the shilling-detection literature."""
    g = df.groupby("user_id")
    dev = (df.rating - df.recipe_id.map(item_mean).fillna(4.4)).abs()
    cnt_i = df.recipe_id.map(pd.Series(base.pop, index=base.items)).fillna(1).clip(lower=1)
    f = pd.DataFrame({
        "length": g.size(),
        "RDMA": (dev / cnt_i).groupby(df.user_id).mean(),
        "WDMA": (dev / cnt_i ** 2).groupby(df.user_id).mean(),
        "mean_rating": g.rating.mean(),
        "rating_std": g.rating.std().fillna(0),
        "share_5_star": (df.rating == 5).groupby(df.user_id).mean(),
        "mean_log_pop": np.log(cnt_i).groupby(df.user_id).mean(),
        "std_log_pop": np.log(cnt_i).groupby(df.user_id).std().fillna(0),
    })
    # DegSim: mean cosine similarity to the 10 most similar genuine users
    Xg = normalize(base.X)
    idx = {r: j for j, r in enumerate(base.items)}
    deg = {}
    for u, items in g.recipe_id:
        v = np.zeros(base.n_items, np.float32)
        v[[idx[i] for i in items if i in idx]] = 1
        s = Xg @ (v / max(np.linalg.norm(v), 1e-9))
        deg[u] = float(np.sort(s)[-11:-1].mean()) if u in base.uid else float(np.sort(s)[-10:].mean())
    f["DegSim"] = pd.Series(deg)
    span = g.date.agg(lambda x: (x.max() - x.min()).days)
    f["active_days"] = span
    return f


genuine_ids = rng.choice(tr.user_id.unique(), 3000, replace=False)
F_gen = features(tr[tr.user_id.isin(genuine_ids)], None).assign(label=0)
F_fake = features(fake_all, None).assign(label=1)
F_fake["target"] = fake_all.groupby("user_id").target.first()
F_fake["kind"] = fake_all.groupby("user_id").kind.first()
# train on attacks against the first 2 targets, test on attacks against the other 2 (unseen targets)
tr_t, te_t = set(targets[:2]), set(targets[2:])
gen_perm = rng.permutation(len(F_gen))
G_tr, G_te = F_gen.iloc[gen_perm[:1500]], F_gen.iloc[gen_perm[1500:]]
A_tr, A_te = F_fake[F_fake.target.isin(tr_t)], F_fake[F_fake.target.isin(te_t)]
det = {}
for name, cols in {"rating features only": ["length", "RDMA", "WDMA", "mean_rating", "rating_std",
                                            "share_5_star", "mean_log_pop", "std_log_pop", "DegSim"],
                   "rating + timing features": ["length", "RDMA", "WDMA", "mean_rating", "rating_std",
                                                "share_5_star", "mean_log_pop", "std_log_pop", "DegSim",
                                                "active_days"]}.items():
    Xtr = pd.concat([G_tr, A_tr])
    Xte = pd.concat([G_te, A_te])
    clf = RandomForestClassifier(300, random_state=SEED, class_weight="balanced").fit(Xtr[cols], Xtr.label)
    pred = clf.predict(Xte[cols])
    per_kind = {k: float(pred[(Xte.label == 1).values & (Xte.get("kind") == k).values].mean())
                for k in ATTACKS}
    det[name] = {"precision": precision_score(Xte.label, pred), "recall": recall_score(Xte.label, pred),
                 "F1": f1_score(Xte.label, pred),
                 "false_positive_rate": float(pred[(Xte.label == 0).values].mean()),
                 "recall_by_attack": per_kind,
                 "importance": dict(sorted(zip(cols, clf.feature_importances_.round(3)), key=lambda x: -x[1]))}
    print(name, det[name], flush=True)
    if name == "rating features only":
        clf_rating, cols_rating = clf, cols
out["detection"] = det

# ------------------------------------------------------------------ defence effect (rating-only detector)
defence = []
for kind in ATTACKS:
    n_fake = int(round(SIZES[-1] * n_genuine))
    hr_before, hr_after = [], []
    for t in targets[2:]:
        fake = make_profiles(kind, t, n_fake, uid0)
        uid0 += n_fake
        Ff = features(fake, None)
        flagged = set(Ff.index[clf_rating.predict(Ff[cols_rating]) == 1])
        Fg = F_gen  # genuine users flagged by mistake would also be removed (measured as FPR above)
        d_att = Data(train=pd.concat([tr, fake]), test=te)
        d_def = Data(train=pd.concat([tr, fake[~fake.user_id.isin(flagged)]]), test=te)
        hr_before.append(hit_ratio(ItemKNN(**iknn_p).fit(d_att), d_att, t))
        hr_after.append(hit_ratio(ItemKNN(**iknn_p).fit(d_def), d_def, t))
    defence.append({"attack": kind, "HR_attacked": float(np.mean(hr_before)), "HR_after_defence": float(np.mean(hr_after))})
    print(defence[-1], flush=True)
out["defence"] = defence
save_json(out, "exp_shilling.json")
