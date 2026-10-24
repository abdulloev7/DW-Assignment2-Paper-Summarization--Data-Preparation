"""Isolation Forest implemented from scratch, following Algorithms 1-3 of
Liu, Ting & Zhou, "Isolation Forest", ICDM 2008."""
import numpy as np

EULER = 0.5772156649


def c(n):
    """Average path length of an unsuccessful BST search over n points (Eq. 1)."""
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    big = n > 2
    out[big] = 2.0 * (np.log(n[big] - 1.0) + EULER) - 2.0 * (n[big] - 1.0) / n[big]
    out[n == 2] = 1.0
    return out


class _ITree:
    """One isolation tree stored as flat arrays (Algorithm 2)."""

    def __init__(self, X, height_limit, rng):
        self.feature, self.threshold, self.left, self.right, self.size = [], [], [], [], []
        self.rng = rng
        self._grow(X, 0, height_limit)
        self.feature = np.array(self.feature)
        self.threshold = np.array(self.threshold)
        self.left = np.array(self.left)
        self.right = np.array(self.right)
        self.size = np.array(self.size)

    def _new_node(self):
        for lst, v in ((self.feature, -1), (self.threshold, 0.0), (self.left, -1),
                       (self.right, -1), (self.size, 0)):
            lst.append(v)
        return len(self.feature) - 1

    def _grow(self, X, e, l):
        node = self._new_node()
        mins, maxs = X.min(axis=0), X.max(axis=0)
        splittable = np.flatnonzero(maxs > mins)
        # stop: height limit reached, one point left, or all points identical
        if e >= l or len(X) <= 1 or len(splittable) == 0:
            self.size[node] = len(X)
            return node
        q = self.rng.choice(splittable)                  # random attribute
        p = self.rng.uniform(mins[q], maxs[q])           # random split value
        mask = X[:, q] < p
        self.feature[node], self.threshold[node] = q, p
        self.left[node] = self._grow(X[mask], e + 1, l)
        self.right[node] = self._grow(X[~mask], e + 1, l)
        return node

    def path_length(self, X):
        """Algorithm 3, vectorised over all instances."""
        node = np.zeros(len(X), dtype=int)
        depth = np.zeros(len(X))
        active = self.feature[node] >= 0
        while active.any():
            idx = np.flatnonzero(active)
            n = node[idx]
            go_left = X[idx, self.feature[n]] < self.threshold[n]
            node[idx] = np.where(go_left, self.left[n], self.right[n])
            depth[idx] += 1
            active = self.feature[node] >= 0
        # external node: add c(size) for the unbuilt subtree
        return depth + c(self.size[node])


class IsolationForestScratch:
    """Algorithm 1: t trees, each grown on a sub-sample of size psi."""

    def __init__(self, n_trees=100, psi=256, random_state=None):
        self.n_trees, self.psi = n_trees, psi
        self.rng = np.random.default_rng(random_state)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.psi_ = min(self.psi, len(X))
        l = int(np.ceil(np.log2(self.psi_)))              # height limit
        self.trees_ = []
        for _ in range(self.n_trees):
            sample = X[self.rng.choice(len(X), self.psi_, replace=False)]
            self.trees_.append(_ITree(sample, l, self.rng))
        return self

    def score(self, X):
        """Anomaly score s(x, psi) = 2^(-E[h(x)] / c(psi)) (Eq. 2); higher = more anomalous."""
        X = np.asarray(X, dtype=float)
        mean_h = np.mean([t.path_length(X) for t in self.trees_], axis=0)
        return 2.0 ** (-mean_h / c(self.psi_))
