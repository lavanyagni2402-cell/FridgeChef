"""Hybrid recommenders - one implementation for each design in the syllabus
(Burke 2002; Jannach et al. 2010):

Monolithic   Feature combination  - content features become pseudo-users in the ALS matrix
             Feature augmentation - item-kNN similarity augmented with content similarity
Parallel     Weighted             - alpha*CF + beta*Item-kNN + gamma*Content
             Switching            - Content for users with few interactions, CF otherwise
             Mixed                - interleave CF and content lists
Pipelined    Cascade              - CF proposes 100 candidates, content re-ranks them
             Meta-level           - content profiles define user similarity for CF
"""
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from models import ALS, ItemKNN, ContentKNN, _topk_sparse
from fridge import minmax


class Weighted:
    name = "Hybrid: Weighted"

    def __init__(self, parts, weights):
        self.parts, self.w = parts, weights

    def fit(self, data):
        return self

    def score(self, users):
        return sum(w * minmax(m.score(users)) for m, w in zip(self.parts, self.w))


class Switching:
    name = "Hybrid: Switching"

    def __init__(self, cold, warm, threshold, counts):
        self.cold, self.warm, self.t, self.counts = cold, warm, threshold, counts

    def fit(self, data):
        return self

    def score(self, users):
        S = minmax(self.warm.score(users))
        cold = self.counts[users] < self.t
        if cold.any():
            S[cold] = minmax(self.cold.score(users[cold]))
        return S


class Mixed:
    """Interleave: positions alternate CF, Content, CF, ... (duplicates skipped)."""
    name = "Hybrid: Mixed"

    def __init__(self, a, b, n_items):
        self.a, self.b, self.n = a, b, n_items

    def fit(self, data):
        return self

    def score(self, users, depth=20):
        Sa, Sb = self.a.score(users), self.b.score(users)
        out = np.full((len(users), self.n), -1e9, np.float32)
        ra = np.argsort(-Sa, 1)[:, :depth * 3]
        rb = np.argsort(-Sb, 1)[:, :depth * 3]
        for r in range(len(users)):
            seen, pos, ia, ib = set(), 0, 0, 0
            while pos < depth * 2:
                src = ra if pos % 2 == 0 else rb
                j = ia if pos % 2 == 0 else ib
                while src[r, j] in seen:
                    j += 1
                seen.add(src[r, j]); out[r, src[r, j]] = depth * 2 - pos
                if pos % 2 == 0:
                    ia = j + 1
                else:
                    ib = j + 1
                pos += 1
        # unseen items keep CF order underneath (so masking of seen items still leaves a full list)
        return out + 1e-6 * minmax(Sa)


class Cascade:
    name = "Hybrid: Cascade"

    def __init__(self, first, second, n_candidates=100, mix=0.5):
        self.first, self.second, self.c, self.mix = first, second, n_candidates, mix

    def fit(self, data):
        return self

    def score(self, users):
        S1 = minmax(self.first.score(users))
        S2 = minmax(self.second.score(users))
        thresh = -np.partition(-S1, self.c, axis=1)[:, self.c][:, None]
        keep = S1 >= thresh
        return np.where(keep, 1 + (1 - self.mix) * S1 + self.mix * S2, S1 * 1e-3)


class FeatureCombination(ALS):
    """Content features are appended as extra 'pseudo-user' rows of the interaction matrix,
    so ALS learns item factors from both behaviour and ingredients/tags."""
    name = "Hybrid: Feature combination"

    def __init__(self, beta=0.3, **kw):
        super().__init__(**kw)
        self.beta = beta

    def fit(self, data):
        cb = ContentKNN().fit(data)
        F = cb.F.T.tocsr().copy()                       # features x items
        F.data = np.ones_like(F.data) * self.beta
        Xa = sp.vstack([data.X, F]).tocsr()
        super().fit(data, X=Xa)
        self.U = self.U[:data.n_users]
        return self


class FeatureAugmentation(ItemKNN):
    """Item-kNN whose CF similarity is augmented with content similarity: S = S_cf + beta*S_content."""
    name = "Hybrid: Feature augmentation"

    def __init__(self, k=100, beta=0.3):
        super().__init__(k)
        self.beta = beta

    def fit(self, data):
        self.X = data.X
        Xc = normalize(data.X.T.tocsr(), axis=1)
        Scf = (Xc @ Xc.T).toarray().astype(np.float32)
        F = ContentKNN().fit(data).F
        Sct = (F @ F.T).toarray().astype(np.float32)
        self.S = _topk_sparse(Scf + self.beta * Sct, self.k).T.tocsr()
        return self


class MetaLevel:
    """Collaboration via content (Pazzani 1999): the content model's learned user profiles
    (not its recommendations) are the input of a user-kNN CF model."""
    name = "Hybrid: Meta-level"

    def __init__(self, k=100):
        self.k = k

    def fit(self, data):
        cb = ContentKNN().fit(data)
        self.P = cb.profiles(data.X, data.R)            # users x features, L2-normalised
        self.X = data.X
        return self

    def score(self, users):
        S = self.P[users] @ self.P.T
        S[np.arange(len(users)), users] = 0
        idx = np.argpartition(-S, self.k, axis=1)[:, self.k:]
        np.put_along_axis(S, idx, 0, axis=1)
        return (sp.csr_matrix(S) @ self.X).toarray()
