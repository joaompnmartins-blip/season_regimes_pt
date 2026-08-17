"""Unit tests for the invariants that must not silently break.

These are deliberately narrow: each one pins a single property that, if broken,
produces plausible-looking but wrong output rather than an exception. That is the
dangerous failure mode in this pipeline - nothing crashes, the maps still look
like weather, and the conclusions are junk.

Run: python tests/test_units.py    (or: pytest tests/test_units.py)
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from regimes_pt import cluster, fire_link, preprocess  # noqa: E402

RNG = np.random.default_rng(42)
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

def test_preprocess():
    print("\npreprocess")

    # Harmonic climatology must absorb an injected annual cycle.
    doy = np.tile(np.arange(1, 366), 8)
    n = len(doy)
    seasonal = 40 * np.cos(2 * np.pi * (doy - 30) / 365.25)
    X = seasonal[:, None] + RNG.normal(0, 1.0, (n, 5))
    clim = preprocess.harmonic_climatology(X, doy, n_harmonics=3)
    resid = X - clim
    check("annual cycle removed", resid.std() < 1.15,
          f"resid sd {resid.std():.3f} vs injected amp 40")

    # Short-window fallback must warn rather than silently misfit.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        preprocess.harmonic_climatology(X[:400], doy[:400] * 0 + 200, n_harmonics=3)
        check("short doy span warns", any(issubclass(x.category, RuntimeWarning)
                                          for x in w))

    # Latitude weights.
    lat = np.array([0.0, 60.0, 89.9])
    w = preprocess.latitude_weights(lat)
    check("sqrt(cos lat) weights", np.allclose(w, np.sqrt(np.cos(np.deg2rad(lat))),
                                               atol=1e-8))
    check("weights non-negative at pole", preprocess.latitude_weights(
        np.array([90.5]))[0] >= 0.0)

    # Domain-mean removal must kill a uniform per-day offset.
    A = RNG.normal(0, 5, (200, 30))
    offset = RNG.normal(0, 20, 200)[:, None]
    aw = np.ones(30)
    out = preprocess.remove_domain_mean(A + offset, aw)
    check("uniform daily offset removed",
          np.allclose(out, preprocess.remove_domain_mean(A, aw), atol=1e-9))

    # Linear detrend must kill a linear trend.
    t = np.arange(300)
    B = RNG.normal(0, 1, (300, 4)) + 0.05 * t[:, None]
    dt = preprocess.linear_detrend(B)
    slope = np.polyfit(t, dt[:, 0], 1)[0]
    check("linear trend removed", abs(slope) < 1e-10, f"residual slope {slope:.2e}")

    # EOF truncation properties.
    C = RNG.normal(0, 1, (400, 25)) @ RNG.normal(0, 1, (25, 25))
    eof = preprocess.eof_truncate(C, var_frac=0.90, max_eofs=30)
    check("explained variance descending",
          np.all(np.diff(eof.explained) <= 1e-12))
    check("var_frac respected", eof.explained.sum() >= 0.90 - 1e-9,
          f"{eof.explained.sum():.4f}")
    check("EOFs orthonormal",
          np.allclose(eof.eofs @ eof.eofs.T, np.eye(eof.n_eof), atol=1e-8))
    check("max_eofs cap honoured",
          preprocess.eof_truncate(C, var_frac=0.999, max_eofs=3).n_eof == 3)

    # Full reconstruction should be near-exact when nothing is truncated.
    full = preprocess.eof_truncate(C, var_frac=1.0, max_eofs=25)
    recon = full.pcs @ full.eofs + C.mean(axis=0)
    check("full-rank reconstruction exact", np.allclose(recon, C, atol=1e-8))

    # PCs must NOT be whitened - leading PCs carry more variance.
    check("PCs not standardised", full.pcs.std(axis=0)[0] > 2 * full.pcs.std(axis=0)[-1])


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------

def test_cluster():
    print("\ncluster")

    P = RNG.normal(0, 1, (4, 60))
    check("self-similarity is 1", abs(cluster.partition_similarity(P, P) - 1.0) < 1e-9)
    check("similarity is symmetric",
          abs(cluster.partition_similarity(P, P[::-1])
              - cluster.partition_similarity(P[::-1], P)) < 1e-9)
    check("permutation invariant",
          abs(cluster.partition_similarity(P, P[[2, 0, 3, 1]]) - 1.0) < 1e-9)

    Q = RNG.normal(0, 1, (4, 60))
    check("unrelated sets score low", cluster.partition_similarity(P, Q) < 0.6,
          f"{cluster.partition_similarity(P, Q):.3f}")

    # A single bad centroid must drag the whole score down (the min is the point).
    Pbad = P.copy()
    Pbad[2] = RNG.normal(0, 1, 60)
    check("one bad cluster penalised",
          cluster.partition_similarity(P, Pbad) < 0.6)

    # AR(1) surrogate preserves lag-1 autocorrelation and variance.
    x = np.zeros((4000, 2))
    for i in range(1, 4000):
        x[i] = 0.8 * x[i - 1] + RNG.normal(0, 1, 2)
    surr = cluster.ar1_surrogate(x, np.random.default_rng(0))
    a_in = np.corrcoef(x[:-1, 0], x[1:, 0])[0, 1]
    a_out = np.corrcoef(surr[:-1, 0], surr[1:, 0])[0, 1]
    check("surrogate preserves lag-1 autocorr", abs(a_in - a_out) < 0.06,
          f"{a_in:.3f} vs {a_out:.3f}")
    check("surrogate preserves variance",
          abs(x[:, 0].std() - surr[:, 0].std()) / x[:, 0].std() < 0.15)

    # Assignment by correlation, with threshold.
    pats = RNG.normal(0, 1, (3, 40))
    days = np.vstack([2.5 * pats[1], -1.0 * pats[0], RNG.normal(0, 1, 40)])
    a = cluster.assign_days(days, pats, corr_threshold=0.4)
    check("amplitude-invariant assignment", a.labels[0] == 1)
    check("sign matters", a.labels[1] != 0)
    check("noise day unclassified", a.labels[2] == -1)
    check("unclassified fraction reported",
          abs(a.unclassified_frac - (a.labels < 0).mean()) < 1e-12)

    # Persistence filter.
    lab = np.array([0, 0, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0])
    filt = cluster.enforce_persistence(lab, min_days=5)
    check("short episodes blanked", filt[0] == -1 and filt[8] == -1)
    check("long episodes kept", np.all(filt[2:8] == 1) and np.all(filt[9:] == 0))

    # Regime statistics.
    check("frequencies sum to classified fraction",
          abs(cluster.regime_frequency(filt, 3).sum() - (filt >= 0).mean()) < 1e-12)
    M = cluster.transition_matrix(np.array([0, 1, 2, 0, 1, 2, 0, 1]), 3)
    check("transition rows normalised",
          np.allclose(M.sum(axis=1), 1.0, atol=1e-9))
    check("self-transitions excluded", np.allclose(np.diag(M), 0.0))


# ---------------------------------------------------------------------------
# fire_link
# ---------------------------------------------------------------------------

def test_fire_link():
    print("\nfire_link")

    # Fixed threshold: ~5% exceedance at p95.
    x = RNG.normal(0, 1, 5000)
    check("p95 gives ~5% exceedance",
          abs(fire_link.exceedance(x, 95.0).mean() - 0.05) < 0.01)

    # Per-doy threshold must neutralise a strong seasonal cycle.
    doy = np.tile(np.arange(152, 274), 40)
    seas = 20 * np.exp(-((doy - 215) / 30.0) ** 2)
    y = seas + RNG.normal(0, 3, len(doy))
    fixed = fire_link.exceedance(y, 95.0)
    perday = fire_link.exceedance(y, 95.0, doy=doy, window=15)
    peak = np.abs(doy - 215) < 20
    check("fixed threshold concentrates on peak season",
          fixed[peak].mean() > 4 * fixed[~peak].mean())
    check("per-doy threshold is seasonally flat",
          abs(perday[peak].mean() - perday[~peak].mean()) < 0.04,
          f"{perday[peak].mean():.3f} vs {perday[~peak].mean():.3f}")

    # Odds ratio against a hand-computed 2x2 table (with 0.5 correction).
    labels = np.array([0] * 100 + [1] * 100)
    exc = np.zeros(200, dtype=bool)
    exc[:40] = True       # 40/100 in regime 0
    exc[100:110] = True   # 10/100 in regime 1
    orr, p_in, p_out = fire_link._odds_ratio(labels, exc, 0)
    expected = (40.5 * 90.5) / (60.5 * 10.5)
    check("odds ratio matches 2x2 table", abs(orr - expected) < 1e-9,
          f"{orr:.4f}")
    check("p_in correct", abs(p_in - 40.5 / 101) < 1e-9)

    # Composite and match score.
    pats = RNG.normal(0, 1, (3, 50))
    anom = np.vstack([pats[2] + RNG.normal(0, 0.2, 50) for _ in range(60)])
    exc2 = np.zeros(60, dtype=bool)
    exc2[:30] = True
    ms = fire_link.match_scores(fire_link.fire_day_composite(anom, exc2), pats)
    check("composite matches source pattern", int(np.argmax(ms)) == 2 and ms[2] > 0.9,
          f"r={ms[2]:.3f}")
    check("empty exceedance raises",
          _raises(fire_link.fire_day_composite, anom, np.zeros(60, dtype=bool)))

    # Block bootstrap must widen intervals on autocorrelated data.
    n = 3000
    lab = np.repeat(RNG.integers(0, 3, n // 10), 10)[:n]
    e = np.zeros(n, dtype=bool)
    p = np.where(lab == 0, 0.20, 0.05)
    e = RNG.random(n) < p
    wide = fire_link.odds_by_regime(lab, e, 3, block_len=10, n_boot=400,
                                    random_state=1)[0]
    narrow = fire_link.odds_by_regime(lab, e, 3, block_len=1, n_boot=400,
                                      random_state=1)[0]
    check("block bootstrap wider than naive",
          (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low),
          f"{wide.ci_high-wide.ci_low:.2f} vs {narrow.ci_high-narrow.ci_low:.2f}")
    check("true effect detected", wide.ci_low > 1.0)

    # Cross-validated skill: no signal must give BSS <= ~0.
    years = np.repeat(np.arange(30), 100)
    rand_lab = RNG.integers(0, 4, 3000)
    rand_exc = RNG.random(3000) < 0.05
    sk = fire_link.cv_brier_skill(rand_lab, rand_exc, years, 4)
    check("no-signal BSS is not positive", sk.bss <= 0.01, f"BSS {sk.bss:+.4f}")

    sig_exc = RNG.random(3000) < np.where(rand_lab == 0, 0.25, 0.03)
    sk2 = fire_link.cv_brier_skill(rand_lab, sig_exc, years, 4)
    check("signal gives positive BSS", sk2.bss > 0.05, f"BSS {sk2.bss:+.4f}")
    check("AUC above chance with signal", sk2.auc > 0.6, f"AUC {sk2.auc:.3f}")

    # Regionalisation keys on co-variability, not mean level.
    t = 400
    shared = RNG.normal(0, 1, t)
    other = RNG.normal(0, 1, t)
    F = np.column_stack([
        shared + RNG.normal(0, .2, t) + 50,    # high mean, group A
        shared + RNG.normal(0, .2, t),         # low mean, group A
        other + RNG.normal(0, .2, t) + 50,     # high mean, group B
        other + RNG.normal(0, .2, t),          # low mean, group B
    ])
    reg = fire_link.regionalize(F, n_regions=2, random_state=0)
    check("regions follow covariability not mean",
          reg[0] == reg[1] and reg[2] == reg[3] and reg[0] != reg[2],
          f"labels {reg.tolist()}")


def _raises(fn, *args) -> bool:
    try:
        fn(*args)
        return False
    except Exception:
        return True


# ---------------------------------------------------------------------------
# regridding (I/O boundary)
# ---------------------------------------------------------------------------

def test_regrid():
    """CADS stopped accepting a server-side `grid` key, so the 0.25 -> 1 deg
    reduction moved into open_z500. It has to stay a block mean: subsampling
    would alias the sub-synoptic detail we are deliberately discarding, and the
    failure would be silent - the maps would still look like weather.

    Skipped rather than failed without xarray, so the numerical core stays
    testable with no I/O stack installed.
    """
    print("\nregrid")
    try:
        import xarray as xr
    except ImportError:
        print("  [SKIP] xarray not installed")
        return

    from regimes_pt.download import _coarsen_to

    lat = np.arange(55.0, 24.9, -0.25)
    lon = np.arange(-40.0, 10.01, 0.25)
    da = xr.DataArray(
        RNG.normal(size=(3, lat.size, lon.size)),
        dims=("time", "latitude", "longitude"),
        coords={"time": [1, 2, 3], "latitude": lat, "longitude": lon},
    )

    c = _coarsen_to(da, 1.0)
    spacing = abs(float(c.latitude[1] - c.latitude[0]))
    check("coarsens to target spacing", abs(spacing - 1.0) < 1e-9,
          f"{spacing:.3f} deg")
    check("16x fewer points at 0.25->1 deg", da.size // c.size == 16,
          f"{da.size} -> {c.size}")

    # Block mean, not a subsample: the first output cell must equal the mean of
    # the 4x4 native block it covers.
    manual = float(da.isel(time=0, latitude=slice(0, 4), longitude=slice(0, 4)).mean())
    got = float(c.isel(time=0, latitude=0, longitude=0))
    check("output cell is the block mean", abs(manual - got) < 1e-10,
          f"{got:.6f} vs {manual:.6f}")

    check("no NaN introduced", bool(np.isfinite(c).all()))
    check("no-op when target is finer than native",
          _coarsen_to(da, 0.25).shape == da.shape)


def main() -> int:
    print("=" * 60)
    print("regimes_pt unit tests")
    print("=" * 60)
    test_preprocess()
    test_cluster()
    test_fire_link()
    test_regrid()
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
        return 1
    print("all unit tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
