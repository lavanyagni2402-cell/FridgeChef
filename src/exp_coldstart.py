"""Experiment 3 - cold start and the switching threshold.
A 'new user' is simulated by hiding all but n of a real user's training interactions
(as if they had just picked n dishes during onboarding). Every model is retrained with
those users truncated; accuracy is measured on their real test-period recipes."""
import json

import numpy as np
import pandas as pd

from common import Data, PROC, RES, SEED, save_json, evaluate_model
from models import Popularity, ItemKNN, ALS, ContentKNN

rng = np.random.default_rng(SEED)
tr = pd.read_pickle(PROC / "train.pkl")
te = pd.read_pickle(PROC / "test.pkl")
cnt = tr.user_id.value_counts()
eligible = cnt[cnt >= 20].index.values
cold_users = rng.choice(eligible, 1500, replace=False)
als_p = json.loads((RES / "tuning_ALS_BPR.json").read_text())["ALS"]["best"]["params"]
iknn_p = json.loads((RES / "tuning_User-kNN_Item-kNN_PureSVD_Content_NeuMF.json").read_text())["Item-kNN"]["best"]["params"]

rows = []
for n in (1, 2, 3, 5, 10, 20):
    is_cold = tr.user_id.isin(cold_users)
    keep = tr[is_cold].groupby("user_id").sample(n, random_state=SEED)
    tr_n = pd.concat([tr[~is_cold], keep])
    d = Data(train=tr_n, test=te)
    users = np.array([d.uid[u] for u in cold_users if d.uid.get(u) in d.truth])
    for m in (Popularity(), ContentKNN(), ItemKNN(**iknn_p), ALS(**als_p)):
        m.fit(d)
        r, _ = evaluate_model(m, d, users=users)
        rows.append({"n_known": n, "model": m.name, **r})
        print(n, m.name, round(r["HR@10"], 4), round(r["NDCG@10"], 4), flush=True)

save_json({"n_cold_users": len(cold_users), "rows": rows}, "exp_coldstart.json")
