"""Export the data the FridgeChef web prototype runs on (app/data.json).

Per recipe: name, time, steps, display ingredients, canonical 'needs' (ids into the ingredient
vocabulary), diet/allergen flags, popularity, mean rating, and two neighbour lists:
  cf - item-kNN cosine on who cooked what (collaborative filtering, 7,452 users)
  ct - TF-IDF cosine on ingredients + tags (content-based)
Plus the ingredient vocabulary and learned ingredient swaps (PPMI + SVD over all 231,637 recipes)."""
import ast
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

from common import Data, ROOT
from models import ContentKNN
from fridge import Substitutes

N_RECIPES = 5000
N_NEIGH = 20
FLAGS = ["has_meat", "has_nuts", "has_dairy", "has_gluten", "has_egg", "has_shellfish", "is_vegetarian", "is_vegan"]

d = Data()
meta = d.item_meta
# subset = most-reviewed recipes in the training period
order = np.argsort(-d.pop)[:N_RECIPES]
sub = meta.iloc[order]
import pandas as pd
raw_steps = pd.read_csv(ROOT / "data" / "RAW_recipes.csv.gz", usecols=["id", "steps", "description"]).set_index("id")

# --- collaborative neighbours (item cosine over all users' cooking history)
Xc = normalize(d.X.T.tocsr()[order], axis=1)
S_cf = (Xc @ Xc.T).toarray()
np.fill_diagonal(S_cf, 0)
# --- content neighbours
cb = ContentKNN().fit(d)
F = cb.F[order]
S_ct = (F @ F.T).toarray()
np.fill_diagonal(S_ct, 0)


def neigh(S):
    idx = np.argsort(-S, axis=1)[:, :N_NEIGH]
    return [[[int(j), round(float(S[i, j]), 3)] for j in row if S[i, j] > 0] for i, row in enumerate(idx)]


cf, ct = neigh(S_cf), neigh(S_ct)

# --- ingredient vocabulary (canonical, non-staple) used by the subset
from collections import Counter
cnt = Counter(x for L in sub["needs"] for x in L)
vocab = [x for x, _ in cnt.most_common()]
vid = {x: i for i, x in enumerate(vocab)}

# --- swaps: mutual top-10 neighbours with score >= 0.4, at most 2
subs = Substitutes(d.recipes["ingredients"].tolist(), min_count=100)
sc = subs.E @ subs.E.T - 0.12 * subs.ppmi
np.fill_diagonal(sc, -1)
sc[subs.together > 0.08] = -1
top10 = np.argsort(-sc, 1)[:, :10]
mutual = np.zeros(sc.shape, bool)
for a in range(len(subs.vocab)):
    mutual[a, top10[a]] = True
mutual &= mutual.T
swaps = {}
for x in vocab:
    if x in subs.idx:
        s = [c for c, v in subs.swaps(x, 6) if v >= 0.4 and mutual[subs.idx[x], subs.idx[c]]][:2]
        if s:
            swaps[vid[x]] = s

# explicit mean rating per recipe (ratings 1-5 only)
tr = d.train[d.train.rating > 0]
mean_r = tr.groupby("recipe_id").rating.mean()

recipes = []
for k, (rid, r) in enumerate(sub.iterrows()):
    flags = sum(1 << b for b, f in enumerate(FLAGS) if bool(r[f]))
    recipes.append({
        "id": int(rid),
        "n": r["name"],
        "m": None if np.isnan(r["minutes"]) else int(r["minutes"]),
        "k": int(r["n_ingredients_clean"]),
        "g": sorted(vid[x] for x in r["needs"]),
        "ri": r["ingredients_raw"],
        "st": ast.literal_eval(raw_steps.at[rid, "steps"]),
        "f": flags,
        "p": int(d.pop[order[k]]),
        "r": round(float(mean_r.get(rid, np.nan)), 2) if rid in mean_r.index else None,
        "c": round(float(r["calories"]), 0),
        "cf": cf[k],
        "ct": ct[k],
    })

out = {"meta": {"n_recipes": len(recipes), "n_users_trained_on": int(d.n_users),
                "n_reviews_trained_on": int(d.X.nnz), "n_swap_recipes": len(d.recipes),
                "flags": FLAGS},
       "vocab": vocab, "vcount": [int(cnt[x]) for x in vocab], "swaps": swaps, "recipes": recipes}
app = ROOT / "app"
app.mkdir(exist_ok=True)
txt = json.dumps(out, separators=(",", ":"), ensure_ascii=False)
(app / "data.json").write_text(txt)
print("recipes", len(recipes), "vocab", len(vocab), "swaps", len(swaps), "MB", round(len(txt) / 1e6, 2))
print("example swaps", {vocab[int(k)]: v for k, v in list(swaps.items())[:25]})
