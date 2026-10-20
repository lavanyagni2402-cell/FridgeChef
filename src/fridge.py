"""FridgeChef knowledge layer: what can you actually cook, and is it allowed?

Kitchen  - recipe x ingredient matrix, pantry coverage, constraint checking,
           constraint relaxation, use-it-up boosting, explanations
Hybrids  - cascade (constraints -> pantry -> taste) and weighted (taste + coverage)
Swaps    - ingredient substitution from PPMI + SVD ingredient embeddings
Critique - case-based "like this, but quicker / fewer ingredients / vegetarian"
"""
import itertools
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from prep import STAPLES

HARD = ("diet", "avoid")                     # never relaxed (safety / ethics)
SOFT_ORDER = ("max_minutes", "max_ingredients", "max_missing", "must_use", "tag")


class Kitchen:
    def __init__(self, meta: pd.DataFrame):
        """meta: recipe rows aligned with model item indices."""
        self.meta = meta
        vocab = sorted({i for L in meta["needs"] for i in L})
        self.ing = {x: j for j, x in enumerate(vocab)}
        self.vocab = vocab
        rows, cols = [], []
        for r, L in enumerate(meta["needs"]):
            for x in L:
                rows.append(r); cols.append(self.ing[x])
        self.M = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(meta), len(vocab)))
        self.n_need = np.asarray(self.M.sum(1)).ravel()
        self.minutes = meta["minutes"].fillna(9999).values
        self.n_ingr = meta["n_ingredients_clean"].values
        self.flags = {c: meta[c].values.astype(bool) for c in meta.columns if c.startswith(("has_", "is_"))}
        self.tags = meta["tags"].apply(set).values

    # ---------------------------------------------------------- pantry
    def pantry_vec(self, pantry):
        v = np.zeros(len(self.vocab), np.float32)
        for x in pantry:
            if x in self.ing:
                v[self.ing[x]] = 1
        return v

    def missing(self, pantry_vecs):
        """pantry_vecs (n_users, n_ing) -> (n_users, n_items) count of missing non-staple ingredients."""
        have = np.asarray((self.M @ pantry_vecs.T).T)
        return self.n_need[None, :] - have

    def coverage(self, missing):
        return 1 - missing / np.maximum(self.n_need[None, :], 1)

    # ---------------------------------------------------------- constraints
    def mask(self, c, missing_row=None, skip=()):
        """Boolean feasibility mask for one user's constraint dict c."""
        m = np.ones(len(self.meta), bool)
        if "diet" in c and "diet" not in skip:
            if c["diet"] == "vegetarian":
                m &= self.flags["is_vegetarian"]
            elif c["diet"] == "vegan":
                m &= self.flags["is_vegan"]
            elif c["diet"] == "no-meat":
                m &= ~self.flags["has_meat"]
        if "avoid" in c and "avoid" not in skip:
            for a in c["avoid"]:
                m &= ~self.flags["has_" + a]
        if "max_minutes" in c and "max_minutes" not in skip:
            m &= self.minutes <= c["max_minutes"]
        if "max_ingredients" in c and "max_ingredients" not in skip:
            m &= self.n_ingr <= c["max_ingredients"]
        if "max_missing" in c and "max_missing" not in skip and missing_row is not None:
            m &= missing_row <= c["max_missing"]
        if "must_use" in c and "must_use" not in skip:
            cols = [self.ing[x] for x in c["must_use"] if x in self.ing]
            if cols:
                m &= np.asarray(self.M[:, cols].sum(1)).ravel() > 0
        if "tag" in c and "tag" not in skip:
            m &= np.array([c["tag"] in t for t in self.tags])
        return m

    def violations(self, items, c):
        """Which hard constraints does each recipe break (for reporting)."""
        ok = self.mask({k: v for k, v in c.items() if k in HARD})
        return ~ok[items]

    def relax(self, c, missing_row=None, k=10):
        """Minimal relaxations: smallest sets of SOFT constraints whose removal gives >= k recipes.
        Hard constraints (diet, allergies) are never offered for relaxation."""
        soft = [s for s in SOFT_ORDER if s in c]
        full = int(self.mask(c, missing_row).sum())
        if full >= k:
            return {"feasible": full, "relaxations": []}
        options = []
        for size in range(1, len(soft) + 1):
            for drop in itertools.combinations(soft, size):
                n = int(self.mask(c, missing_row, skip=drop).sum())
                if n >= k:
                    options.append({"drop": list(drop), "results": n})
            if options:
                break
        return {"feasible": full, "relaxations": sorted(options, key=lambda o: -o["results"])}

    # ---------------------------------------------------------- explanations
    def explain(self, item, pantry_vec, liked_items=None, F=None):
        need = [self.vocab[j] for j in self.M[item].indices]
        have = [x for x in need if pantry_vec[self.ing[x]]]
        miss = [x for x in need if not pantry_vec[self.ing[x]]]
        out = {"recipe": self.meta.iloc[item]["name"], "you_have": f"{len(have)}/{len(need)}",
               "missing": miss, "minutes": self.meta.iloc[item]["minutes"]}
        if liked_items is not None and F is not None and len(liked_items):
            sims = F[liked_items] @ F[item].T
            sims = np.asarray(sims.toarray() if sp.issparse(sims) else sims).ravel()
            out["because_you_liked"] = self.meta.iloc[liked_items[int(np.argmax(sims))]]["name"]
        return out


def minmax(S):
    lo, hi = S.min(1, keepdims=True), S.max(1, keepdims=True)
    return (S - lo) / np.maximum(hi - lo, 1e-9)


def cascade_rank(taste, feasible, k=10):
    """Pipelined hybrid: infeasible recipes are removed BEFORE taste ranking."""
    s = np.where(feasible, taste, -np.inf)
    return np.argsort(-s, axis=1)[:, :k]


def weighted_rank(taste, coverage, alpha, seen_mask=None, k=10):
    """Parallel weighted hybrid: alpha * taste + (1 - alpha) * pantry coverage."""
    s = alpha * minmax(taste) + (1 - alpha) * coverage
    if seen_mask is not None:
        s = np.where(seen_mask, -np.inf, s)
    return np.argsort(-s, axis=1)[:, :k]


# -------------------------------------------------------------- ingredient swaps
class Substitutes:
    """Ingredient embeddings from PPMI co-occurrence over ALL recipes, reduced with SVD.
    Two ingredients are good swaps if they appear in similar contexts (high embedding
    similarity) but rarely together in the same recipe (you don't use butter AND margarine)."""

    def __init__(self, ingredient_lists, min_count=100, dims=100):
        cnt = Counter(i for L in ingredient_lists for i in set(L) if i not in STAPLES)
        self.vocab = sorted([i for i, c in cnt.items() if c >= min_count])
        self.idx = {x: j for j, x in enumerate(self.vocab)}
        rows, cols = [], []
        for r, L in enumerate(ingredient_lists):
            js = {self.idx[i] for i in L if i in self.idx}
            rows += [r] * len(js); cols += list(js)
        B = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ingredient_lists), len(self.vocab)))
        C = (B.T @ B).toarray()
        self.count = np.diag(C).copy()
        np.fill_diagonal(C, 0)
        total = len(ingredient_lists)
        pij = C / total
        pi = self.count / total
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(pij / np.outer(pi, pi))
        ppmi = np.nan_to_num(np.maximum(pmi, 0), neginf=0, posinf=0)
        self.ppmi = ppmi
        U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
        self.E = normalize(U[:, :dims] * np.sqrt(S[:dims]))
        # how often two ingredients appear together, relative to the rarer one
        self.together = C / np.maximum(np.minimum.outer(self.count, self.count), 1)

    def swaps(self, ing, n=5, max_together=0.08, penalty=0.12, min_score=0.3):
        """Score = context similarity - penalty * direct PPMI. Substitutes share neighbours but
        rarely share a recipe; complements (mozzarella + zucchini) share recipes and are pushed down.
        Variants of the same ingredient ('dark brown sugar' for 'brown sugar') are skipped."""
        if ing not in self.idx:
            return []
        j = self.idx[ing]
        score = self.E @ self.E[j] - penalty * self.ppmi[j]
        score[j] = -1
        score[self.together[j] > max_together] = -1
        words = set(ing.split())
        out = []
        for b in np.argsort(-score):
            if score[b] < min_score or len(out) == n:
                break
            cand = self.vocab[b]
            cw = set(cand.split())
            if words <= cw or cw <= words or ing in cand or cand in ing:
                continue                                  # a variant, not a swap
            if any(set(c.split()) <= cw or cw <= set(c.split()) for c, _ in out):
                continue                                  # near-duplicate of a swap already listed
            out.append((cand, round(float(score[b]), 3)))
        return out


# -------------------------------------------------------------- case-based critiquing
CRITIQUES = {
    "quicker": lambda K, ref: K.minutes < K.minutes[ref] * 0.75,
    "fewer ingredients": lambda K, ref: K.n_ingr <= K.n_ingr[ref] - 2,
    "vegetarian": lambda K, ref: K.flags["is_vegetarian"],
    "lighter": lambda K, ref: K.meta["calories"].values < K.meta["calories"].values[ref] * 0.75,
    "no dairy": lambda K, ref: ~K.flags["has_dairy"],
}


def critique(K, F, ref, crit, n=3):
    """Return the recipes most similar to `ref` (content similarity) that satisfy the critique."""
    sim = F @ F[ref].T
    sim = np.asarray(sim.toarray() if sp.issparse(sim) else sim).ravel()
    ok = CRITIQUES[crit](K, ref)
    sim = np.where(ok, sim, -1)
    sim[ref] = -1
    best = np.argsort(-sim)[:n]
    return [(K.meta.iloc[b]["name"], round(float(sim[b]), 3), int(K.minutes[b]) if K.minutes[b] < 9999 else None,
             int(K.n_ingr[b])) for b in best]
