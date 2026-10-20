"""Recommendation models for FridgeChef.

Every model exposes   fit(data) -> self   and   score(user_indices) -> (n, n_items) array.
Collaborative:  Popularity, UserKNN, ItemKNN, PureSVD, ALS (implicit), BPR, NeuMF (neural CF)
Content-based:  ContentKNN (TF-IDF item profiles over ingredients + tags)
"""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

from common import SEED

GENERIC_TAGS = {"time-to-make", "course", "preparation", "main-ingredient", "dietary", "occasion",
                "cuisine", "equipment", "number-of-servings", "easy", "taste-mood",
                "60-minutes-or-less", "30-minutes-or-less", "4-hours-or-less",
                "15-minutes-or-less", "3-steps-or-less", "5-ingredients-or-less",
                "1-day-or-more", "technique", "low-in-something", "high-in-something",
                "to-go", "inexpensive", "seasonal", "holiday-event", "for-1-or-2",
                "north-american", "american", "beginner-cook", "novelty"}


def _topk_sparse(S, k):
    """Keep the k largest entries per row of a dense similarity matrix -> csr."""
    S = np.asarray(S)
    np.fill_diagonal(S, 0)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    rows = np.repeat(np.arange(S.shape[0]), k)
    vals = np.take_along_axis(S, idx, 1).ravel()
    return sp.csr_matrix((vals, (rows, idx.ravel())), shape=S.shape)


class Popularity:
    name = "Popularity"

    def fit(self, data):
        self.p = data.pop.astype(np.float32)
        return self

    def score(self, users):
        return np.tile(self.p, (len(users), 1))


class UserKNN:
    """User-based nearest neighbours, cosine similarity on the implicit matrix."""
    name = "User-kNN"

    def __init__(self, k=50):
        self.k = k

    def fit(self, data):
        self.X = data.X
        Xn = normalize(data.X, axis=1)
        self.Xn = Xn
        return self

    def score(self, users):
        S = (self.Xn[users] @ self.Xn.T).toarray()
        S[np.arange(len(users)), users] = 0
        idx = np.argpartition(-S, self.k, axis=1)[:, self.k:]
        np.put_along_axis(S, idx, 0, axis=1)           # keep only k neighbours
        return (sp.csr_matrix(S) @ self.X).toarray()


class ItemKNN:
    """Item-based nearest neighbours, cosine similarity between item columns."""
    name = "Item-kNN"

    def __init__(self, k=100):
        self.k = k

    def fit(self, data):
        self.X = data.X
        Xc = normalize(data.X.T.tocsr(), axis=1)
        S = (Xc @ Xc.T).toarray().astype(np.float32)
        self.S = _topk_sparse(S, self.k).T.tocsr()     # column j = neighbours of j
        return self

    def score(self, users):
        return (self.X[users] @ self.S).toarray()


class PureSVD:
    """Truncated SVD of the user-item matrix (Cremonesi et al., 2010)."""
    name = "PureSVD"

    def __init__(self, f=50):
        self.f = f

    def fit(self, data):
        svd = TruncatedSVD(self.f, random_state=SEED)
        self.U = svd.fit_transform(data.X)
        self.V = svd.components_
        return self

    def score(self, users):
        return self.U[users] @ self.V


class ALS:
    """Implicit-feedback ALS (Hu, Koren & Volinsky, 2008) via the `implicit` library."""
    name = "ALS (implicit MF)"

    def __init__(self, f=64, reg=0.05, alpha=20.0, iters=20):
        self.kw = dict(factors=f, regularization=reg, alpha=alpha, iterations=iters, random_state=SEED)

    def fit(self, data, X=None):
        from implicit.als import AlternatingLeastSquares
        self.m = AlternatingLeastSquares(**self.kw)
        self.m.fit((data.X if X is None else X).tocsr(), show_progress=False)
        self.U, self.V = np.asarray(self.m.user_factors), np.asarray(self.m.item_factors)
        return self

    def fold_in(self, user_items_csr):
        """Compute factors for new users from their (few) interactions, model frozen."""
        U = self.m.recalculate_user(np.arange(user_items_csr.shape[0]), user_items_csr)
        return np.asarray(U)

    def score(self, users):
        return self.U[users] @ self.V.T


class BPR:
    """Bayesian Personalised Ranking MF (Rendle et al., 2009)."""
    name = "BPR-MF"

    def __init__(self, f=64, lr=0.05, reg=0.01, iters=100):
        self.kw = dict(factors=f, learning_rate=lr, regularization=reg, iterations=iters, random_state=SEED)

    def fit(self, data):
        from implicit.bpr import BayesianPersonalizedRanking
        m = BayesianPersonalizedRanking(**self.kw)
        m.fit(data.X.tocsr(), show_progress=False)
        self.U, self.V = np.asarray(m.user_factors), np.asarray(m.item_factors)
        return self

    def score(self, users):
        return self.U[users] @ self.V.T


class NeuMF:
    """Neural Collaborative Filtering (He et al., 2017): GMF + MLP towers, BCE with negatives."""
    name = "NeuMF (neural CF)"

    def __init__(self, f=32, layers=(64, 32, 16), neg=4, epochs=8, lr=1e-3, batch=2048):
        self.f, self.layers, self.neg, self.epochs, self.lr, self.batch = f, layers, neg, epochs, lr, batch

    def fit(self, data):
        import torch
        import torch.nn as nn
        torch.manual_seed(SEED)
        torch.set_num_threads(2)
        nu, ni = data.n_users, data.n_items
        f, L = self.f, self.layers

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.ug, s.ig = nn.Embedding(nu, f), nn.Embedding(ni, f)
                s.um, s.im = nn.Embedding(nu, L[0] // 2), nn.Embedding(ni, L[0] // 2)
                mlp = []
                for a, b in zip(L[:-1], L[1:]):
                    mlp += [nn.Linear(a, b), nn.ReLU()]
                s.mlp = nn.Sequential(*mlp)
                s.out = nn.Linear(f + L[-1], 1)
                for e in (s.ug, s.ig, s.um, s.im):
                    nn.init.normal_(e.weight, std=0.01)

            def forward(s, u, i):
                g = s.ug(u) * s.ig(i)
                m = s.mlp(torch.cat([s.um(u), s.im(i)], -1))
                return s.out(torch.cat([g, m], -1)).squeeze(-1)

        self.net = Net()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        lossf = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(SEED)
        coo = data.X.tocoo()
        pu, pi = coo.row, coo.col
        seen = data.X.tocsr()
        self.history = []
        for ep in range(self.epochs):
            nu_ = np.repeat(pu, self.neg)
            ni_ = rng.integers(0, ni, len(nu_))
            # resample negatives that collide with positives (rare at 0.3% density)
            clash = np.asarray(seen[nu_, ni_]).ravel() > 0
            ni_[clash] = rng.integers(0, ni, clash.sum())
            U = np.concatenate([pu, nu_]); I = np.concatenate([pi, ni_])
            Y = np.concatenate([np.ones(len(pu)), np.zeros(len(nu_))]).astype(np.float32)
            perm = rng.permutation(len(U))
            U, I, Y = torch.tensor(U[perm]), torch.tensor(I[perm]), torch.tensor(Y[perm])
            tot = 0.0
            self.net.train()
            for s in range(0, len(U), self.batch):
                opt.zero_grad()
                loss = lossf(self.net(U[s:s + self.batch], I[s:s + self.batch]), Y[s:s + self.batch])
                loss.backward(); opt.step()
                tot += loss.item() * len(Y[s:s + self.batch])
            self.history.append(tot / len(U))
        self.net.eval()
        self.ni = ni
        return self

    def score(self, users):
        import torch
        out = []
        items = torch.arange(self.ni)
        with torch.no_grad():
            for u in users:
                out.append(self.net(torch.full((self.ni,), int(u)), items).numpy())
        return np.vstack(out)


def item_documents(meta):
    """Item profile = canonical ingredients + informative tags (Food.com user-entered tags)."""
    docs = []
    for ings, tags in zip(meta["ingredients"], meta["tags"]):
        d = ["ing:" + i.replace(" ", "_") for i in ings]
        d += ["tag:" + t for t in tags if t not in GENERIC_TAGS]
        docs.append(d)
    return docs


class ContentKNN:
    """Content-based: TF-IDF item profiles, user profile = rating-weighted mean of liked items."""
    name = "Content-based (TF-IDF)"

    def __init__(self, svd_dims=None):
        self.svd_dims = svd_dims
        if svd_dims:
            self.name = f"Content-based (TF-IDF + SVD {svd_dims})"

    def fit(self, data, X=None):
        self.vec = TfidfVectorizer(analyzer=lambda d: d, min_df=3, sublinear_tf=True)
        F = self.vec.fit_transform(item_documents(data.item_meta))
        if self.svd_dims:
            self.svd = TruncatedSVD(self.svd_dims, random_state=SEED)
            F = self.svd.fit_transform(F)
        self.F = normalize(F)
        self.R = data.R
        self.X = data.X if X is None else X
        return self

    def profiles(self, X_rows, R_rows=None):
        W = X_rows.copy().astype(np.float32)
        if R_rows is not None:          # liked (4-5) -> weight 2, disliked (1-2) -> -0.5, else 1
            L = R_rows.copy().astype(np.float32)
            L.data = np.where(L.data >= 4, 1.0, np.where(L.data <= 2, -1.5, 0.0)).astype(np.float32)
            W = W + L
        P = W @ self.F
        return normalize(np.asarray(P.toarray() if sp.issparse(P) else P))

    def score(self, users, X_rows=None):
        Xr = self.X[users] if X_rows is None else X_rows
        Rr = self.R[users] if X_rows is None else None
        P = self.profiles(Xr, Rr)
        if sp.issparse(self.F):
            return np.asarray((self.F @ P.T).T)
        return P @ self.F.T
