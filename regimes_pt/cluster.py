"""Weather-regime clustering and the choice of k.

The number of regimes is not a fact about the atmosphere - it is a choice, and
different domains and seasons support different choices. The standard defence is
the classifiability index of Michelangeli et al. (1995): a partition into k
clusters is admissible only if repeated k-means runs converge on *the same*
k centroids more consistently than they would on red-noise data with identical
autocorrelation and variance spectrum.

That test is the whole point of this module. Without it, k=4 is an assumption
imported from the winter literature; with it, k=4 either survives on your data
or it does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Core fitting
# ---------------------------------------------------------------------------

def fit_kmeans(pcs: np.ndarray, k: int, n_init: int = 100,
               random_state: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, centroids_in_pc_space)."""
    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(pcs)
    return labels, km.cluster_centers_


def centroids_to_patterns(centroids_pc: np.ndarray, eofs: np.ndarray) -> np.ndarray:
    """Project PC-space centroids back onto the (weighted) spatial grid."""
    return centroids_pc @ eofs


# ---------------------------------------------------------------------------
# Classifiability index
# ---------------------------------------------------------------------------

def _pattern_corr_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise correlation between rows of A and rows of B."""
    Ac = A - A.mean(axis=1, keepdims=True)
    Bc = B - B.mean(axis=1, keepdims=True)
    Ac /= np.linalg.norm(Ac, axis=1, keepdims=True)
    Bc /= np.linalg.norm(Bc, axis=1, keepdims=True)
    return Ac @ Bc.T


def partition_similarity(P: np.ndarray, Q: np.ndarray) -> float:
    """Michelangeli similarity between two centroid sets.

    For every centroid of P take its best match in Q; the similarity of the pair
    of partitions is the *worst* of those best matches. Taking the minimum is
    what makes the index strict: one unreproducible cluster is enough to
    disqualify the whole partition.
    """
    C = _pattern_corr_matrix(P, Q)
    return float(min(C.max(axis=1).min(), C.max(axis=0).min()))


def classifiability_index(
    pcs: np.ndarray,
    k: int,
    n_partitions: int = 50,
    n_init: int = 10,
    random_state: int = 0,
) -> float:
    """Mean pairwise partition similarity over `n_partitions` k-means runs."""
    rng = np.random.default_rng(random_state)
    sets = []
    for _ in range(n_partitions):
        seed = int(rng.integers(0, 2**31 - 1))
        _, cent = fit_kmeans(pcs, k, n_init=n_init, random_state=seed)
        sets.append(cent)

    sims = [
        partition_similarity(sets[i], sets[j])
        for i in range(len(sets))
        for j in range(i + 1, len(sets))
    ]
    return float(np.mean(sims))


def ar1_surrogate(pcs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Red-noise surrogate preserving each PC's variance and lag-1 autocorrelation.

    PCs are uncorrelated by construction, so generating them independently is
    the right null: it keeps the persistence and the variance spectrum but
    destroys any genuine multi-modality.
    """
    n, m = pcs.shape
    out = np.empty((n, m))
    for j in range(m):
        x = pcs[:, j]
        a = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        a = np.clip(a, -0.99, 0.99)
        s = x.std() * np.sqrt(max(1.0 - a ** 2, 1e-6))
        e = rng.normal(0.0, s, size=n)
        y = np.empty(n)
        y[0] = rng.normal(0.0, x.std())
        for i in range(1, n):
            y[i] = a * y[i - 1] + e[i]
        out[:, j] = y
    return out


@dataclass
class KSelection:
    k: int
    ci: float
    ci_null_p90: float
    ci_null_p95: float
    significant: bool


def select_k(
    pcs: np.ndarray,
    k_range: Sequence[int] = (2, 3, 4, 5, 6, 7, 8, 9, 10),
    n_partitions: int = 50,
    n_surrogates: int = 100,
    n_init: int = 10,
    n_partitions_surrogate: int = 20,
    random_state: int = 0,
    verbose: bool = True,
) -> list[KSelection]:
    """Classifiability index vs a red-noise null for each candidate k.

    This is the expensive step: cost scales as
    n_surrogates * n_partitions_surrogate * n_init * len(k_range).
    Defaults are tuned to run in minutes on a 5,500-day JJAS archive; raise
    n_surrogates to 500 for a publication-grade null.
    """
    rng = np.random.default_rng(random_state)
    results = []

    for k in k_range:
        ci = classifiability_index(
            pcs, k, n_partitions=n_partitions, n_init=n_init,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )

        null = []
        for _ in range(n_surrogates):
            surr = ar1_surrogate(pcs, rng)
            null.append(classifiability_index(
                surr, k, n_partitions=n_partitions_surrogate, n_init=n_init,
                random_state=int(rng.integers(0, 2**31 - 1)),
            ))
        null = np.asarray(null)
        p90, p95 = np.percentile(null, [90, 95])

        res = KSelection(k=k, ci=ci, ci_null_p90=float(p90),
                         ci_null_p95=float(p95), significant=bool(ci > p95))
        results.append(res)
        if verbose:
            flag = "SIGNIFICANT" if res.significant else "-"
            print(f"  k={k:2d}  CI={ci:.4f}  null p95={p95:.4f}  {flag}")

    return results


# ---------------------------------------------------------------------------
# Assignment of days to regimes
# ---------------------------------------------------------------------------

@dataclass
class Assignment:
    labels: np.ndarray          # (time,) regime index, -1 where unclassified
    max_corr: np.ndarray        # (time,) spatial correlation with assigned centroid
    unclassified_frac: float


def assign_days(
    anom_w: np.ndarray,
    patterns_w: np.ndarray,
    corr_threshold: float = 0.40,
) -> Assignment:
    """Assign each day to its most-correlated regime pattern.

    Correlation rather than Euclidean distance is used deliberately: it makes the
    assignment sensitive to the *shape* of the flow anomaly and not its
    amplitude, which is what regime membership is supposed to mean. Days whose
    best correlation falls below `corr_threshold` are labelled -1.

    Expect a substantial unclassified fraction, especially in summer when
    anomalies are weak. That is a real property of the atmosphere, not a bug -
    forcing every day into a regime is how these classifications get oversold.
    """
    C = _pattern_corr_matrix(anom_w, patterns_w)
    labels = C.argmax(axis=1)
    best = C.max(axis=1)
    labels = np.where(best >= corr_threshold, labels, -1)
    return Assignment(
        labels=labels,
        max_corr=best,
        unclassified_frac=float((labels < 0).mean()),
    )


def project_and_assign(
    field: np.ndarray, lat: np.ndarray, lon: np.ndarray,
    clim: np.ndarray, area_w: np.ndarray,
    patterns_w: np.ndarray, corr_threshold: float = 0.40,
) -> Assignment:
    """Assign *new* days (e.g. forecast fields) to previously fitted centroids.

    The climatology and weights must be the ones derived from the training
    archive - recomputing them on a short forecast window is a classic and
    silent source of leakage.
    """
    nt = field.shape[0]
    X = field.reshape(nt, -1).astype(float)
    anom = X - clim
    anom = anom - (anom @ (area_w / area_w.sum()))[:, None]
    return assign_days(anom * area_w, patterns_w, corr_threshold=corr_threshold)


# ---------------------------------------------------------------------------
# Regime statistics
# ---------------------------------------------------------------------------

def enforce_persistence(labels: np.ndarray, min_days: int = 5) -> np.ndarray:
    """Blank out regime episodes shorter than `min_days`.

    Optional. Useful when the downstream product is a multi-day pre-positioning
    decision, where a two-day excursion into a regime is operationally
    meaningless.
    """
    out = labels.copy()
    n = len(labels)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and labels[j + 1] == labels[i]:
            j += 1
        if labels[i] >= 0 and (j - i + 1) < min_days:
            out[i:j + 1] = -1
        i = j + 1
    return out


def regime_frequency(labels: np.ndarray, k: int) -> np.ndarray:
    """Fraction of all days in each regime (excludes unclassified from the count)."""
    return np.array([(labels == i).mean() for i in range(k)])


def mean_persistence(labels: np.ndarray, k: int) -> np.ndarray:
    """Mean length in days of consecutive episodes of each regime."""
    out = np.zeros(k)
    for r in range(k):
        runs, cur = [], 0
        for v in labels:
            if v == r:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        out[r] = float(np.mean(runs)) if runs else 0.0
    return out


def transition_matrix(labels: np.ndarray, k: int,
                      exclude_self: bool = True) -> np.ndarray:
    """Row-normalised day-to-day transition probabilities between regimes."""
    M = np.zeros((k, k))
    for a, b in zip(labels[:-1], labels[1:]):
        if a >= 0 and b >= 0:
            M[a, b] += 1
    if exclude_self:
        np.fill_diagonal(M, 0.0)
    rs = M.sum(axis=1, keepdims=True)
    return np.divide(M, rs, out=np.zeros_like(M), where=rs > 0)
