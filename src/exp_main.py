"""Experiment 2 - top-10 ranking on the temporal test set: all base models + all hybrid designs.
Hybrid weights are chosen on the validation split (from train only), then applied once to test."""
import itertools
import json
import time

import numpy as np

from common import Data, RES, SEED, save_json, evaluate_model
from models import Popularity, UserKNN, ItemKNN, PureSVD, ALS, BPR, NeuMF, ContentKNN
from hybrids import Weighted, Mixed, Cascade, FeatureCombination, FeatureAugmentation, MetaLevel
from tune import validation_data

T1 = json.loads((RES / "tuning_User-kNN_Item-kNN_PureSVD_Content_NeuMF.json").read_text())
T2 = json.loads((RES / "tuning_ALS_BPR.json").read_text())
bp = lambda T, n: T[n]["best"]["params"]
P = {"User-kNN": bp(T1, "User-kNN"), "Item-kNN": bp(T1, "Item-kNN"), "PureSVD": bp(T1, "PureSVD"),
     "Content": bp(T1, "Content"), "NeuMF": bp(T1, "NeuMF"), "ALS": bp(T2, "ALS"), "BPR": bp(T2, "BPR")}
CF = {"User-kNN": UserKNN, "Item-kNN": ItemKNN, "PureSVD": PureSVD, "ALS": ALS}
best_cf = max(CF, key=lambda n: (T1 | T2)[n]["best"]["NDCG@10"])
print("best CF on validation:", best_cf, P, flush=True)


def build_hybrids(d, cfg):
    cf = CF[best_cf](**P[best_cf]).fit(d)
    ik = ItemKNN(**P["Item-kNN"]).fit(d)
    cb = ContentKNN().fit(d)
    al = ALS(**P["ALS"]).fit(d)
    hy = {
        "Hybrid: Weighted": Weighted([cf, ik, cb] if best_cf != "Item-kNN" else [cf, al, cb], cfg["w"]),
        "Hybrid: Mixed": Mixed(cf, cb, d.n_items),
        "Hybrid: Cascade": Cascade(cf, cb, 100, cfg["mix"]),
        "Hybrid: Feature combination": FeatureCombination(beta=cfg["fc_beta"], **{k: v for k, v in P["ALS"].items()}).fit(d),
        "Hybrid: Feature augmentation": FeatureAugmentation(k=P["Item-kNN"]["k"], beta=cfg["fa_beta"]).fit(d),
        "Hybrid: Meta-level": MetaLevel(k=P["User-kNN"]["k"]).fit(d),
    }
    return hy


# ---------------------------------------------------------------- choose hybrid settings on validation
v = validation_data()
vu = np.random.default_rng(SEED).choice(v.test_users, 2000, replace=False)
cf_v = CF[best_cf](**P[best_cf]).fit(v)
second = ItemKNN(**P["Item-kNN"]).fit(v) if best_cf != "Item-kNN" else ALS(**P["ALS"]).fit(v)
cb_v = ContentKNN().fit(v)
tuned = {}
best = (-1, None)
for w in itertools.product((1.0,), (0.0, 0.25, 0.5, 1.0), (0.0, 0.1, 0.25, 0.5)):
    r, _ = evaluate_model(Weighted([cf_v, second, cb_v], w), v, users=vu)
    best = max(best, (r["NDCG@10"], w))
tuned["w"] = best[1]
best = (-1, None)
for mix in (0.1, 0.25, 0.5):
    r, _ = evaluate_model(Cascade(cf_v, cb_v, 100, mix), v, users=vu)
    best = max(best, (r["NDCG@10"], mix))
tuned["mix"] = best[1]
best = (-1, None)
for b in (0.05, 0.2, 0.5):
    r, _ = evaluate_model(FeatureCombination(beta=b, **P["ALS"]).fit(v), v, users=vu)
    best = max(best, (r["NDCG@10"], b))
tuned["fc_beta"] = best[1]
best = (-1, None)
for b in (0.1, 0.3, 1.0):
    r, _ = evaluate_model(FeatureAugmentation(k=P["Item-kNN"]["k"], beta=b).fit(v), v, users=vu)
    best = max(best, (r["NDCG@10"], b))
tuned["fa_beta"] = best[1]
print("hybrid settings", tuned, flush=True)

# ---------------------------------------------------------------- final test evaluation
d = Data()
rows, tops = [], {}
base = [Popularity(), UserKNN(**P["User-kNN"]), ItemKNN(**P["Item-kNN"]), PureSVD(**P["PureSVD"]),
        ALS(**P["ALS"]), BPR(**P["BPR"]), NeuMF(**P["NeuMF"]), ContentKNN(**P["Content"])]
for m in base:
    t = time.time(); m.fit(d)
    r, top = evaluate_model(m, d)
    rows.append({"model": m.name, "family": "Content-based" if "Content" in m.name else ("Baseline" if m.name == "Popularity" else "Collaborative"),
                 "secs": round(time.time() - t, 1), **r})
    tops[m.name] = top
    print(rows[-1], flush=True)
    if isinstance(m, NeuMF):
        neumf_loss = m.history
for name, m in build_hybrids(d, tuned).items():
    t = time.time()
    r, top = evaluate_model(m, d)
    rows.append({"model": name, "family": "Hybrid", "secs": round(time.time() - t, 1), **r})
    tops[name] = top
    print(rows[-1], flush=True)

# overlap between the two most different base models -> argument for hybridisation
a, b = tops[[k for k in tops if k.startswith("ALS")][0]], tops[[k for k in tops if k.startswith("Content")][0]]
overlap = float(np.mean([len(set(x) & set(y)) / 10 for x, y in zip(a, b)]))
save_json({"best_cf": best_cf, "params": P, "hybrid_settings": tuned, "rows": rows,
           "top10_overlap_ALS_vs_content": overlap, "neumf_train_loss": neumf_loss,
           "n_test_users": len(d.test_users)}, "exp_main.json")
