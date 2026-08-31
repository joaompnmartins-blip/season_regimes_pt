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


def stratified_rate(labels: np.ndarray, strata: np.ndarray,
                    exceed: np.ndarray, k: int, n_strata: int,
                    min_days: int = 20) -> np.ndarray:
    """Exceedance rate in every (stratum, regime) cell, as a fraction.

    The point of the stratification is to ask whether the regime still says
    anything once a stronger predictor has spoken. Stratifying on FWI quartile
    and finding the regimes still spread within a stratum is the only evidence
    that the circulation adds information rather than restating the fuel state.

    Cells thinner than `min_days` come back NaN rather than as a rate computed
    from a handful of days, which would be noise wearing the shape of a signal.
    """
    out = np.full((n_strata, k), np.nan)
    for s in range(n_strata):
        for r in range(k):
            cell = (strata == s) & (labels == r)
            if cell.sum() >= min_days:
                out[s, r] = float(exceed[cell].mean())
    return out


def episodes(labels: np.ndarray, regime: int) -> np.ndarray:
    """Lengths of consecutive runs of `regime`, in days.

    Regimes persist for about a week, and that persistence is what makes them
    plannable: a decision taken against a 4-day block is actionable in a way a
    single flagged day is not. Callers use the run-length distribution to state
    how many decision windows a season actually offers.

    Runs are counted within the array as given, so a caller passing several
    seasons concatenated must accept that a run spanning a season boundary is
    an artefact - JJAS archives have no such boundary days adjacent in reality.
    """
    lengths = []
    run = 0
    for x in labels:
        if x == regime:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
    return np.asarray(lengths, dtype=int)


@dataclass
class StandDownResult:
    regime: int
    days: int                   # days spent in the regime
    day_fraction: float         # share of the season released
    events_missed: int          # exceedance days inside the regime
    event_fraction: float       # share of all exceedance days released
    burned_fraction: float      # share of total burned area released


def stand_down(labels: np.ndarray, exceed: np.ndarray, burned: np.ndarray,
               regime: int) -> StandDownResult:
    """What standing down on every day of `regime` would have cost.

    Deliberately a hindcast with no forecast uncertainty in it: this is the
    ceiling on what a regime-conditioned stand-down can achieve, never the
    expectation. Presenting it as anything else would overstate the product by
    exactly the amount of skill the forecast turns out to lack.
    """
    inr = labels == regime
    total_burn = float(burned.sum())
    return StandDownResult(
        regime=regime,
        days=int(inr.sum()),
        day_fraction=float(inr.mean()),
        events_missed=int(exceed[inr].sum()),
        event_fraction=float(exceed[inr].sum() / max(exceed.sum(), 1)),
        burned_fraction=float(burned[inr].sum() / total_burn) if total_burn else 0.0,
    )


def degrade_labels(labels: np.ndarray, alpha: float,
                   rng: np.random.Generator) -> np.ndarray:
    """A synthetic regime forecast that is right with probability `alpha`.

    Answers the question that decides whether any of this is operational,
    without needing an S2S archive: instead of asking what skill a forecast has
    at week 3, ask how much skill the signal *requires*, then compare that
    against published skill.

    When wrong, the forecast falls back to the climatological distribution
    rather than to a uniform draw. That is how a forecast actually degrades -
    towards climatology, not towards nonsense - and it means the fallback is
    sometimes right by luck, so `alpha` is always below the realized hit rate.
    Report the realized rate, since that is what published skill scores measure.
    """
    cats, counts = np.unique(labels, return_counts=True)
    clim = rng.choice(cats, size=len(labels), p=counts / counts.sum())
    return np.where(rng.random(len(labels)) < alpha, labels, clim)


def degrade_series(x: np.ndarray, rho: float,
                   rng: np.random.Generator) -> np.ndarray:
    """A synthetic forecast of a continuous predictor with correlation `rho`.

    The counterpart of `degrade_labels` for a weekly regime *frequency*, which
    is the quantity sub-seasonal forecasts are actually scored on. A daily
    categorical label and a weekly frequency are different products with
    different skill, and conflating them flatters the weaker one.
    """
    z = (x - x.mean()) / x.std()
    return rho * z + np.sqrt(max(1.0 - rho ** 2, 0.0)) * rng.standard_normal(len(z))


def block_aggregate(values: np.ndarray, season: np.ndarray, block_len: int = 7):
    """Reduce a daily series to non-overlapping blocks that never span seasons.

    A JJAS archive puts 30 September next to 1 June of the following year, so a
    naive rolling window silently averages across a nine-month gap. Blocks are
    cut within each season and the ragged tail of each season is dropped.
    """
    idx = []
    for s in np.unique(season):
        where = np.flatnonzero(season == s)
        for i in range(0, len(where) - block_len + 1, block_len):
            idx.append(where[i:i + block_len])
    return np.asarray(idx, dtype=int)


@dataclass
class LeadResult:
    horizon: int                # window is t+1 .. t+horizon
    regime: int
    stratum: int                # -1 when unstratified
    rate_in: float              # mean exceedance rate in the window, after the regime
    rate_out: float             # ... after every other day
    ratio: float
    ci_low: float
    ci_high: float
    n_in: int                   # origin days in the regime


def forward_window(values: np.ndarray, season: np.ndarray, horizon: int):
    """Mean of `values` over t+1..t+horizon, per origin day t.

    Returns `(origins, window_mean)` where `origins` indexes the day the
    window is issued from. Windows that would run past the end of a season are
    dropped rather than wrapped, for the reason given in `block_aggregate`:
    consecutive rows of a JJAS archive straddle a nine-month gap at the year
    boundary, and a window that spans it is meteorological nonsense.

    Assumes days within one season are contiguous and in date order, which is
    what the pipeline produces; a gappy series would silently shorten the
    effective horizon.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least one day")
    origins, means = [], []
    for s in np.unique(season):
        where = np.flatnonzero(season == s)
        for i in range(len(where) - horizon):
            origins.append(where[i])
            means.append(values[where[i + 1:i + 1 + horizon]].mean())
    return np.asarray(origins, dtype=int), np.asarray(means, dtype=float)


def _rate_ratio(inr: np.ndarray, wmean: np.ndarray) -> float:
    if inr.sum() == 0 or (~inr).sum() == 0:
        return np.nan
    out = wmean[~inr].mean()
    return float(wmean[inr].mean() / out) if out > 0 else np.nan


def _block_ci(inr: np.ndarray, wmean: np.ndarray, block_len: int,
              n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    n = len(inr)
    n_blocks = int(np.ceil(n / block_len))
    pool = np.arange(0, max(n - block_len + 1, 1))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.choice(pool, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
        idx = idx[idx < n]
        boots[b] = _rate_ratio(inr[idx], wmean[idx])
    if np.all(np.isnan(boots)):
        return float("nan"), float("nan")
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def lead_ratio(labels: np.ndarray, exceed: np.ndarray, season: np.ndarray,
               regime: int, horizon: int, block_len: int = 7,
               n_boot: int = 2000, random_state: int = 0) -> LeadResult:
    """Exceedance rate over the next `horizon` days, given today's regime.

    The lead this measures is *free*: the conditioning day is analysed, not
    forecast, so unlike everything in `step_forecast` there is no skill
    requirement to discount. Whatever survives here survives operationally.

    Use this rather than a single lagged odds ratio at lead L. A planner
    commits resources across a window and cannot act on "day 4 specifically";
    the window mean is the quantity the decision is actually taken on, and it
    is the more favourable of the two because the near days carry most of the
    signal.
    """
    origins, wmean = forward_window(exceed.astype(float), season, horizon)
    inr = labels[origins] == regime
    rng = np.random.default_rng(random_state)
    lo, hi = _block_ci(inr, wmean, block_len, n_boot, rng)
    return LeadResult(
        horizon=horizon, regime=regime, stratum=-1,
        rate_in=float(wmean[inr].mean()) if inr.any() else float("nan"),
        rate_out=float(wmean[~inr].mean()) if (~inr).any() else float("nan"),
        ratio=_rate_ratio(inr, wmean), ci_low=lo, ci_high=hi,
        n_in=int(inr.sum()),
    )


def lead_ratio_by_stratum(labels: np.ndarray, exceed: np.ndarray,
                          covariate: np.ndarray, season: np.ndarray,
                          regime: int, horizon: int, n_strata: int = 4,
                          min_days: int = 15, block_len: int = 7,
                          n_boot: int = 2000,
                          random_state: int = 0) -> list[LeadResult]:
    """`lead_ratio` within quantile bins of a window-mean covariate.

    The covariate is averaged over the same forward window as the outcome, not
    taken at the origin day, because the intended covariate is FWI and the
    comparison of interest is against what an FWI *forecast* would already
    tell a planner about those days. Stratifying on the origin-day value
    instead would understate how much of the regime signal FWI subsumes.

    This is the test that decides whether the regime layer earns its place:
    ICNF already runs on fire danger, so a regime signal that vanishes inside
    an FWI stratum is a repackaging of FWI rather than an addition to it.
    Strata thinner than `min_days` regime days are returned with NaN interval
    rather than dropped, so the caller can see the coverage.
    """
    origins, wmean = forward_window(exceed.astype(float), season, horizon)
    _, cmean = forward_window(np.asarray(covariate, dtype=float), season, horizon)
    inr = labels[origins] == regime
    edges = np.quantile(cmean, np.linspace(0, 1, n_strata + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    rng = np.random.default_rng(random_state)

    results = []
    for j in range(n_strata):
        m = (cmean >= edges[j]) & (cmean < edges[j + 1])
        a, w = inr[m], wmean[m]
        if a.sum() < min_days or (~a).sum() < min_days:
            lo = hi = float("nan")
        else:
            lo, hi = _block_ci(a, w, block_len, n_boot, rng)
        results.append(LeadResult(
            horizon=horizon, regime=regime, stratum=j,
            rate_in=float(w[a].mean()) if a.any() else float("nan"),
            rate_out=float(w[~a].mean()) if (~a).any() else float("nan"),
            ratio=_rate_ratio(a, w), ci_low=lo, ci_high=hi, n_in=int(a.sum()),
        ))
    return results


def compare_configurations(results: Sequence[tuple[str, SkillResult]]) -> str:
    """Format a comparison table of candidate configurations."""
    lines = [f"{'configuration':<28}{'BSS':>9}{'AUC':>8}{'n':>8}", "-" * 53]
    for name, s in sorted(results, key=lambda t: -t[1].bss):
        lines.append(f"{name:<28}{s.bss:>9.4f}{s.auc:>8.3f}{s.n:>8d}")
    return "\n".join(lines)
