"""Experiment 4 - the FridgeChef layer: pantry-aware ranking, hard constraints,
constraint relaxation, use-it-up mode, ingredient swaps and critiquing.

Pantry protocol: Food.com has no fridge contents, so each user's pantry is INFERRED as the
25 non-staple ingredients they used most often in their training recipes (staples such as
salt, pepper and water are always assumed available). Accuracy is still measured against the
recipes the user really cooked in the test period."""
import json
from collections import Counter

import numpy as np

from common import Data, RES, SEED, save_json, ranking_metrics
from models import ALS, ContentKNN
from fridge import Kitchen, cascade_rank, weighted_rank, minmax, Substitutes, critique

PANTRY_SIZE = 25
rng = np.random.default_rng(SEED)


def best_params(name, file):
    return json.loads((RES / file).read_text())[name]["best"]["params"]


d = Data()
K = Kitchen(d.item_meta)
als = ALS(**best_params("ALS", "tuning_ALS_BPR.json")).fit(d)
content = ContentKNN().fit(d)
users = d.test_users
truth = [d.truth[u] for u in users]
out = {"pantry_size": PANTRY_SIZE, "n_users": len(users)}

# ---------------------------------------------------------------- inferred pantries
pantries = []
for u in users:
    items = d.X[u].indices
    c = Counter(x for L in d.item_meta["needs"].iloc[items] for x in L)
    pantries.append([x for x, _ in c.most_common(PANTRY_SIZE)])
P = np.vstack([K.pantry_vec(p) for p in pantries])
miss = K.missing(P)                                   # users x items
cov = K.coverage(miss)
seen = d.X[users].toarray() > 0
taste = als.score(users)
taste_masked = np.where(seen, -np.inf, taste)

# ceiling: how many of the recipes users really cooked next were cookable from the pantry?
ceil = {}
for m in (0, 1, 2, 3, 5):
    ok = [np.mean([miss[r, i] <= m for i in t]) for r, t in enumerate(truth)]
    ceil[m] = float(np.mean(ok))
out["share_of_true_next_recipes_cookable"] = ceil
out["catalogue_cookable_share"] = {m: float((miss <= m).mean()) for m in (0, 1, 2, 3, 5)}


def describe(top):
    r = ranking_metrics(top, truth, d.n_items, d.pop, d.n_users)
    mm = np.take_along_axis(miss, top, 1)
    r["avg_missing"] = float(mm.mean())
    r["pct_cook_now"] = float((mm == 0).mean() * 100)
    return r


rows = []
top = np.argsort(-taste_masked, 1)[:, :10]
rows.append({"method": "Taste only (ALS)", "m": None, **describe(top)})
fr = np.where(seen, -np.inf, cov + 1e-3 * minmax(np.log1p(d.pop)[None, :]))
rows.append({"method": "Fridge only (coverage)", "m": None, **describe(np.argsort(-fr, 1)[:, :10])})
for m in (0, 1, 2, 3):
    top = cascade_rank(taste_masked, miss <= m)
    rows.append({"method": "Cascade (filter, then taste)", "m": m, **describe(top)})
for a in (0.2, 0.4, 0.6, 0.8, 0.9):
    top = weighted_rank(np.where(seen, taste.min(), taste), cov, a, seen_mask=seen)
    rows.append({"method": f"Weighted (alpha={a})", "m": None, "alpha": a, **describe(top)})
out["pantry_ranking"] = rows
for r in rows:
    print(r["method"], r.get("m"), {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, flush=True)

# ---------------------------------------------------------------- hard constraints (personas)
personas = {
    "Vegetarian + nut allergy": {"diet": "vegetarian", "avoid": ["nuts"]},
    "Gluten-free + dairy-free": {"avoid": ["gluten", "dairy"]},
    "Vegan": {"diet": "vegan"},
    "Egg + shellfish allergy": {"avoid": ["egg", "shellfish"]},
}
sample = rng.choice(len(users), 2000, replace=False)
pers = {}
for pname, c in personas.items():
    ok = K.mask(c)
    plain = np.argsort(-taste_masked[sample], 1)[:, :10]
    viol = ~ok[plain]
    res = {"catalogue_share_allowed": float(ok.mean()),
           "taste_only_violating_share_of_top10": float(viol.mean()),
           "taste_only_users_with_any_violation": float(viol.any(1).mean())}
    for lam in (0.1, 0.3, 1.0):
        s = minmax(np.where(seen[sample], taste.min(), taste[sample])) - lam * (~ok)[None, :]
        s = np.where(seen[sample], -np.inf, s)
        t = np.argsort(-s, 1)[:, :10]
        res[f"soft_penalty_{lam}_violating_share"] = float((~ok[t]).mean())
    t = cascade_rank(taste_masked[sample], ok[None, :] & np.ones((len(sample), 1), bool))
    res["cascade_violating_share"] = float((~ok[t]).mean())
    pers[pname] = res
    print(pname, res, flush=True)
out["personas"] = pers

# ---------------------------------------------------------------- constraint relaxation
soft = {"max_minutes": 30, "max_ingredients": 8, "max_missing": 1}
relax_stats = Counter()
need_relax = 0
examples = []
for pname, c in personas.items():
    for j in sample[:500]:
        cc = {**c, **soft}
        r = K.relax(cc, missing_row=miss[j], k=10)
        if r["relaxations"]:
            need_relax += 1
            relax_stats[tuple(r["relaxations"][0]["drop"])] += 1
            if len(examples) < 4 and pname not in [e["persona"] for e in examples]:
                examples.append({"persona": pname, "constraints": cc, **r})
        elif r["feasible"] < 10:
            relax_stats[("no soft relaxation enough",)] += 1
out["relaxation"] = {"cases": 2000, "needed_relaxation": need_relax,
                     "most_common_fix": {" + ".join(k): v for k, v in relax_stats.most_common()},
                     "examples": examples}
print(out["relaxation"], flush=True)

# ---------------------------------------------------------------- use-it-up mode
uiu = []
exp_users = sample[:1000]
expiring = []
for j in exp_users:
    have = np.flatnonzero(P[j])
    expiring.append(rng.choice(have, 2, replace=False))
uses = np.vstack([np.asarray(K.M[:, e].sum(1)).ravel() > 0 for e in expiring])
for beta in (0.0, 0.1, 0.25, 0.5, 1.0):
    s = minmax(np.where(seen[exp_users], taste.min(), taste[exp_users])) + beta * uses
    s = np.where(seen[exp_users] | (miss[exp_users] > 2), -np.inf, s)
    t = np.argsort(-s, 1)[:, :10]
    m = ranking_metrics(t, [truth[j] for j in exp_users], d.n_items, d.pop, d.n_users)
    uiu.append({"beta": beta, "pct_top10_using_expiring": float(np.take_along_axis(uses, t, 1).mean() * 100),
                "HR@10": m["HR@10"], "NDCG@10": m["NDCG@10"]})
    print(uiu[-1], flush=True)
out["use_it_up"] = uiu

# ---------------------------------------------------------------- ingredient swaps (all 231k recipes)
subs = Substitutes(d.recipes["ingredients"].tolist())
probe = ["buttermilk", "butter", "sour cream", "heavy cream", "brown sugar", "lemon juice", "honey",
         "chicken broth", "mozzarella cheese", "ricotta cheese", "shallot", "walnut", "white wine",
         "mayonnaise", "spinach", "basil", "red wine vinegar", "egg", "breadcrumb", "maple syrup"]
out["swaps"] = {p: subs.swaps(p, 4) for p in probe}
for p in probe:
    print("swap", p, out["swaps"][p])

# ---------------------------------------------------------------- critiquing + explanation
popular = np.argsort(-d.pop)[:200]
refs = [i for i in popular if K.minutes[i] >= 40 and K.n_ingr[i] >= 8][:3]
out["critique_examples"] = []
for ref in refs:
    ex = {"reference": d.item_meta.iloc[ref]["name"], "minutes": int(K.minutes[ref]), "ingredients": int(K.n_ingr[ref])}
    for crit in ("quicker", "fewer ingredients", "vegetarian", "lighter"):
        ex[crit] = critique(K, content.F, ref, crit)
    out["critique_examples"].append(ex)
    print(ex)

j = sample[0]
top = cascade_rank(taste_masked[[j]], (miss[[j]] <= 1))[0][:5]
liked = d.X[users[j]].indices
out["explanation_example"] = {"pantry": pantries[j],
                              "recs": [K.explain(i, P[j], liked, content.F) for i in top]}
print(out["explanation_example"])
save_json(out, "exp_fridge.json")
