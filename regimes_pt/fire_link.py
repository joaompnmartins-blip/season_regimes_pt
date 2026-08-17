"""Linking regimes to Portuguese fire weather.

This module contains the actual test of the Portugal hypothesis. The question is
not "do regimes exist" - they do, in any dataset - but "does any regime centroid
actually resemble the circulation on Portugal's extreme fire-weather days, and
does knowing the regime change the odds of such a day".

Two diagnostics answer it:

* `fire_day_composite` + `match_scores` - build the mean Z500 anomaly on the
  worst FWI days for a region and ask how well the best centroid correlates with
  it. Canonical k=4 is expected to score poorly if the Iberian pattern really is
  sub-synoptic; a Portugal-tuned partition should score higher. That difference
  is the result worth having.

* `cv_brier_skill` - the objective, out-of-sample comparison between candidate
  configurations. Leave-one-year-out, so a configuration cannot win by
  memorising its own training years.

Everything here uses a *block* bootstrap. Daily fire weather is strongly
autocorrelated and regimes persist for a week at a time, so treating days as
independent inflates the effective sample size by roughly an order of magnitude
and produces confidence intervals that are dramatically too narrow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Regionalisation
# ---------------------------------------------------------------------------

def regionalize(
    fwi: np.ndarray,             # (time, n_points) FWI at each land grid point
    n_regions: int = 6,
    random_state: int = 0,
) -> np.ndarray:
    """Cluster grid points into homogeneous fire-climate zones.

    Grid points are standardised in time first, so the clustering keys on the
    *co-variability* of fire weather rather than on the mean level. Otherwise
    you simply recover a north-south gradient of climatological FWI, which you
    already knew.

    Returns a (n_points,) integer array of region labels. Prefer this over
    distritos: administrative boundaries cut straight across the Litoral/Interior
    gradient that actually organises Portuguese fire weather.
    """
    from sklearn.cluster import KMeans

    X = fwi.T.astype(float)                       # (points, time)
    X = X - X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    X = np.divide(X, sd, out=np.zeros_like(X), where=sd > 0)

    km = KMeans(n_clusters=n_regions, n_init=50, random_state=random_state)
    return km.fit_predict(X)


def region_series(fwi: np.ndarray, region_labels: np.ndarray) -> np.ndarray:
    """Area-mean FWI series per region -> (time, n_regions)."""
    ids = np.unique(region_labels)
    return np.column_stack([fwi[:, region_labels == r].mean(axis=1) for r in ids])


def exceedance(
    series: np.ndarray,
    percentile: float = 95.0,
    doy: Optional[np.ndarray] = None,
    window: int = 15,
) -> np.ndarray:
    """Boolean exceedance of a percentile threshold.

    If `doy` is supplied the threshold is computed per calendar day from a
    +/- `window` day pool, which removes the within-season cycle. Without it,
    a fixed seasonal threshold will make August look far more dangerous than
    June purely by construction - fine if that is what you want operationally,
    misleading if you are trying to isolate the circulation signal.
    """
    x = np.asarray(series, dtype=float)
    if doy is None:
        return x >= np.percentile(x, percentile)

    doy = np.asarray(doy)
    out = np.zeros(len(x), dtype=bool)
    for d in np.unique(doy):
        sel = np.abs(((doy - d + 182) % 365) - 182) <= window
        thr = np.percentile(x[sel], percentile)
        out[doy == d] = x[doy == d] >= thr
    return out


# ---------------------------------------------------------------------------
# Composite diagnostics
# ---------------------------------------------------------------------------

def fire_day_composite(anom_w: np.ndarray, exceed: np.ndarray) -> np.ndarray:
    """Mean weighted Z500 anomaly over the flagged extreme fire-weather days."""
    if exceed.sum() == 0:
        raise ValueError("no exceedance days flagged")
    return anom_w[exceed].mean(axis=0)


def match_scores(composite: np.ndarray, patterns_w: np.ndarray) -> np.ndarray:
    """Spatial correlation of the fire-day composite with each regime centroid.

    Interpretation: max score < ~0.5 means no regime in this partition captures
    the fire-day circulation, and regime-conditioned guidance for that region
    will be weak no matter how the probabilities are dressed up.
    """
    c = composite - composite.mean()
    P = patterns_w - patterns_w.mean(axis=1, keepdims=True)
    return (P @ c) / (np.linalg.norm(P, axis=1) * np.linalg.norm(c))


# ---------------------------------------------------------------------------
# Odds ratios with block bootstrap
# ---------------------------------------------------------------------------

@dataclass
class OddsResult:
    regime: int
    p_in: float                 # P(exceed | regime)
    p_out: float                # P(exceed | not regime)
    odds_ratio: float
    ci_low: float
    ci_high: float
    n_days: int


def _odds_ratio(labels: np.ndarray, exceed: np.ndarray, r: int,
                correction: float = 0.5) -> tuple[float, float, float]:
    inr = labels == r
    a = float((exceed & inr).sum()) + correction
    b = float((~exceed & inr).sum()) + correction
    c = float((exceed & ~inr).sum()) + correction
    d = float((~exceed & ~inr).sum()) + correction
    return (a * d) / (b * c), a / (a + b), c / (c + d)


def odds_by_regime(
    labels: np.ndarray,
    exceed: np.ndarray,
    k: int,
    block_len: int = 7,
    n_boot: int = 2000,
    random_state: int = 0,
) -> list[OddsResult]:
    """Regime-conditioned odds ratios with moving-block-bootstrap intervals.

    `block_len` should be of the order of the regime persistence (about a week
    in summer). Setting it to 1 recovers the naive independent-days interval,
    which will be far too narrow - useful only to see how large the difference is.
    """
    rng = np.random.default_rng(random_state)
    n = len(labels)
    n_blocks = int(np.ceil(n / block_len))
    starts_pool = np.arange(0, max(n - block_len + 1, 1))

    boots = np.empty((n_boot, k))
    for b in range(n_boot):
        starts = rng.choice(starts_pool, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
        idx = idx[idx < n]
        for r in range(k):
            boots[b, r] = _odds_ratio(labels[idx], exceed[idx], r)[0]

    results = []
    for r in range(k):
        orr, p_in, p_out = _odds_ratio(labels, exceed, r)
        lo, hi = np.nanpercentile(boots[:, r], [2.5, 97.5])
        results.append(OddsResult(
            regime=r, p_in=p_in, p_out=p_out, odds_ratio=orr,
            ci_low=float(lo), ci_high=float(hi), n_days=int((labels == r).sum()),
        ))
    return results


# ---------------------------------------------------------------------------
# Out-of-sample skill
# ---------------------------------------------------------------------------

@dataclass
class SkillResult:
    brier: float
    brier_clim: float
    bss: float
    auc: float
    n: int


def cv_brier_skill(
    labels: np.ndarray,
    exceed: np.ndarray,
    years: np.ndarray,
    k: int,
) -> SkillResult:
    """Leave-one-year-out skill of the regime-conditioned climatological forecast.

    The 'forecast' is deliberately trivial: given today's regime, issue
    P(exceed | that regime) estimated from every year except this one. It is a
    floor, not a product - but it is exactly the right quantity for comparing
    candidate domain/k configurations, because any difference in BSS is
    attributable to the regime definition alone.

    A BSS near zero means the regime carries no information about extreme fire
    weather in that region beyond climatology, whatever the composite maps look
    like.
    """
    years = np.asarray(years)
    p = np.full(len(labels), np.nan)

    for y in np.unique(years):
        tr, te = years != y, years == y
        base = exceed[tr].mean()
        rates = np.array([
            exceed[tr & (labels == r)].mean() if (tr & (labels == r)).sum() >= 10
            else base
            for r in range(k)
        ])
        lab_te = labels[te]
        p[te] = np.where(lab_te >= 0, rates[np.clip(lab_te, 0, k - 1)], base)

    ok = ~np.isnan(p)
    p, y_true = p[ok], exceed[ok].astype(float)

    brier = float(np.mean((p - y_true) ** 2))
    clim = float(np.mean((y_true.mean() - y_true) ** 2))
    bss = 1.0 - brier / clim if clim > 0 else np.nan

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, p)) if 0 < y_true.sum() < len(y_true) else np.nan
    except Exception:
        auc = np.nan

    return SkillResult(brier=brier, brier_clim=clim, bss=float(bss),
                       auc=auc, n=int(ok.sum()))


def compare_configurations(results: Sequence[tuple[str, SkillResult]]) -> str:
    """Format a comparison table of candidate configurations."""
    lines = [f"{'configuration':<28}{'BSS':>9}{'AUC':>8}{'n':>8}", "-" * 53]
    for name, s in sorted(results, key=lambda t: -t[1].bss):
        lines.append(f"{name:<28}{s.bss:>9.4f}{s.auc:>8.3f}{s.n:>8d}")
    return "\n".join(lines)
