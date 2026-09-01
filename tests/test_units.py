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


# ---------------------------------------------------------------------------
# download planning (no network)
# ---------------------------------------------------------------------------

def test_download_plan():
    """Which years get requested is pure bookkeeping, but getting it wrong is
    expensive and quiet: skip a year that is missing and the archive has a hole
    no later step reports; refetch one that is present and an overnight run
    wastes a queue slot per mistake.
    """
    print("\ndownload plan")
    import os
    import tempfile

    from regimes_pt.download import _chunk, _years_on_disk

    with tempfile.TemporaryDirectory() as d:
        for fname in ("z500_pt_tuned_JJAS_2003.nc",
                      "z500_pt_tuned_JJAS_1980-1989.nc",
                      "z500_pt_tuned_JJAS_notayear.nc"):
            open(os.path.join(d, fname), "w").close()
        found = _years_on_disk(os.path.join(d, "z500_pt_tuned_JJAS_*.nc"))

    check("single-year filename parsed", 2003 in found)
    check("year-range filename expanded", found.issuperset(range(1980, 1990)),
          f"{len(found & set(range(1980, 1990)))}/10 of 1980-1989")
    check("range is inclusive of its end", 1989 in found)
    check("unparseable filename ignored", 1990 not in found and 2004 not in found)

    # Chunking must partition exactly - no dropped or duplicated years.
    want = [y for y in range(1980, 2026) if y not in found]
    chunks = list(_chunk(want, 10))
    flat = [y for c in chunks for y in c]
    check("chunks partition the wanted years", flat == want,
          f"{len(chunks)} chunks, {len(flat)} years")
    check("no chunk exceeds the request size", all(len(c) <= 10 for c in chunks))
    check("already-present years excluded", 2003 not in flat and 1985 not in flat)


def test_multifile():
    """Chunk files can interleave, and the reader must sort rather than raise.

    Skipping years already on disk produces a file whose span encloses another
    file's - 2001-2007 holding only 2001, 2002 and 2007, around a separate 2003
    file. open_mfdataset(combine="by_coords") raises on that, and it only shows
    up once such a chunk exists, which was after the full download finished.
    """
    print("\nmultifile read")
    try:
        import xarray as xr
    except ImportError:
        print("  [SKIP] xarray not installed")
        return

    import os
    import tempfile

    from regimes_pt.download import _open_many

    def block(years):
        t = np.concatenate([
            np.arange(f"{y}-06-01", f"{y}-06-06", dtype="datetime64[D]")
            for y in years])
        return xr.Dataset(
            {"z": ("valid_time", np.arange(len(t), dtype="float64"))},
            coords={"valid_time": t.astype("datetime64[ns]")})

    with tempfile.TemporaryDirectory() as d:
        # Deliberately interleaved: the enclosing span is written first.
        paths = []
        for name, years in (("a_2001-2007", [2001, 2002, 2007]),
                            ("b_2003", [2003]),
                            ("c_2004-2006", [2004, 2005, 2006])):
            p = os.path.join(d, name + ".nc")
            block(years).to_netcdf(p)
            paths.append(p)
        ds = _open_many(paths)

    t = ds["time"].values
    check("interleaved spans open without raising", len(t) == 35, f"{len(t)} days")
    check("time axis is sorted", bool((np.diff(t) > np.timedelta64(0)).all()))
    check("no days lost or duplicated",
          len(np.unique(t)) == 35 and len(t) == 35)
    years = sorted({int(str(x)[:4]) for x in t})
    check("every year present once", years == [2001, 2002, 2003, 2004, 2005,
                                               2006, 2007], f"{years}")


# ---------------------------------------------------------------------------
# fire layer
# ---------------------------------------------------------------------------

def test_fire_layer():
    """The occurrence layer's three claims, each of which fails silently.

    A stratified rate computed from a thin cell, a run-length that merges two
    episodes, or a stand-down fraction taken against the wrong denominator all
    produce numbers that look operational and are wrong.
    """
    print("\nfire layer")

    # Stratification must not invent rates from thin cells.
    lab = np.array([0, 0, 0, 1, 1, 1] * 20)
    strata = np.array([0] * 60 + [1] * 60)
    exc = np.zeros(120, dtype=bool)
    exc[strata == 1] = True                      # every event in stratum 1
    rates = fire_link.stratified_rate(lab, strata, exc, k=2, n_strata=2,
                                      min_days=20)
    check("stratified rate recovers the cell rate",
          np.allclose(rates[0], 0.0) and np.allclose(rates[1], 1.0))
    thin = fire_link.stratified_rate(lab, strata, exc, k=2, n_strata=2,
                                     min_days=100)
    check("thin cells return NaN, not a rate", bool(np.isnan(thin).all()))

    # Run lengths: the persistence claim rests on these.
    runs = fire_link.episodes(np.array([0, 0, 1, 0, 0, 0, 1, 1, 0]), 0)
    check("episode lengths correct", list(runs) == [2, 3, 1], f"{list(runs)}")
    check("a run touching the end is counted",
          list(fire_link.episodes(np.array([1, 0, 0]), 0)) == [2])
    check("absent regime gives no episodes",
          len(fire_link.episodes(np.array([1, 1, 1]), 0)) == 0)

    # Stand-down accounting: fractions must use the right denominators.
    lab = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    exc = np.array([1, 0, 0, 0, 1, 1, 1, 1], dtype=bool)
    burned = np.array([10.0, 0, 0, 0, 100.0, 100.0, 100.0, 100.0])
    sd = fire_link.stand_down(lab, exc, burned, regime=0)
    check("stand-down day fraction", abs(sd.day_fraction - 0.5) < 1e-12)
    check("events missed counted inside the regime", sd.events_missed == 1)
    check("event fraction is of ALL events, not of the regime",
          abs(sd.event_fraction - 0.2) < 1e-12, f"{sd.event_fraction:.3f}")
    check("burned fraction is of ALL burned area",
          abs(sd.burned_fraction - 10.0 / 410.0) < 1e-12)

    # The asymmetry the proposal rests on: releasing many days should be able
    # to cost few events. If this ever inverts, the headline claim is dead.
    check("a large day share can carry a small event share",
          sd.day_fraction > 2 * sd.event_fraction,
          f"{sd.day_fraction:.2f} days vs {sd.event_fraction:.2f} events")


# ---------------------------------------------------------------------------
# forecast degradation
# ---------------------------------------------------------------------------

def test_forecast_penalty():
    """The synthetic-forecast machinery that decides whether this is operational.

    Each property here, if wrong, flatters the forecast: a degradation that
    keeps too much signal, a correlation that overshoots, or a weekly block
    that spans the nine-month gap between two JJAS seasons.
    """
    print("\nforecast penalty")
    rng = np.random.default_rng(0)

    lab = np.repeat(np.arange(4), 250)
    check("alpha=1 returns the truth unchanged",
          np.array_equal(fire_link.degrade_labels(lab, 1.0, rng), lab))

    # Falling back to climatology means the fallback is sometimes right by
    # luck, so realized accuracy must sit ABOVE alpha, never below it.
    acc = [float(np.mean(fire_link.degrade_labels(lab, a, rng) == lab))
           for a in (0.0, 0.25, 0.5)]
    check("realized accuracy exceeds alpha (lucky climatology hits)",
          all(m > a - 1e-9 for m, a in zip(acc, (0.0, 0.25, 0.5))),
          f"{[round(x, 3) for x in acc]}")
    check("alpha=0 lands near the climatological hit rate",
          abs(acc[0] - 0.25) < 0.05, f"{acc[0]:.3f} vs 1/4")

    # A degraded continuous forecast must actually hit its target correlation.
    x = rng.standard_normal(4000)
    for rho in (0.3, 0.6, 0.9):
        got = np.corrcoef(x, fire_link.degrade_series(x, rho, rng))[0, 1]
        check(f"degrade_series achieves rho={rho}", abs(got - rho) < 0.05,
              f"got {got:.3f}")

    # Blocks must never straddle a season boundary.
    season = np.repeat([2001, 2002], 10)
    blocks = fire_link.block_aggregate(np.arange(20), season, block_len=7)
    check("no block spans two seasons",
          all(len(set(season[b])) == 1 for b in blocks))
    check("blocks are exactly block_len long",
          all(len(b) == 7 for b in blocks))
    check("ragged season tail is dropped, not padded", len(blocks) == 2,
          f"{len(blocks)} blocks from 2 seasons of 10 days")


def test_free_lead():
    """The forward-window machinery behind `--step lead`.

    The failure that matters here is a window that reaches across the
    nine-month gap between two JJAS seasons: it would not raise, it would just
    quietly average September fire days into a June forecast and inflate the
    lead the signal appears to carry.
    """
    print("\nfree lead")

    # Two 10-day seasons laid end to end, as the archive stores them.
    season = np.repeat([2001, 2002], 10)
    values = np.concatenate([np.zeros(10), np.ones(10)])

    origins, wmean = fire_link.forward_window(values, season, 3)
    check("forward window never spans two seasons",
          set(np.unique(wmean)) == {0.0, 1.0},
          f"means seen: {sorted(set(wmean.tolist()))}")
    check("forward window drops the ragged tail of each season",
          len(origins) == 2 * (10 - 3),
          f"{len(origins)} origins from 2 seasons of 10 days, horizon 3")
    check("window is strictly forward-looking",
          origins[0] == 0 and wmean[0] == 0.0)

    # A regime that genuinely suppresses the next few days must come back with
    # a ratio below 1; the point of the test is that the sign cannot silently
    # flip through an off-by-one in the window offset.
    rng = np.random.default_rng(3)
    n_seasons, n_days = 40, 60
    season = np.repeat(np.arange(n_seasons), n_days)
    labels = rng.integers(0, 3, n_seasons * n_days)
    # Suppress - but only partially. Zeroing the days after a regime-0 day
    # gives an exact 0.0 ratio with a degenerate interval, which would let a
    # broken bootstrap pass the width check below.
    hazard = np.full(n_seasons * n_days, 0.25)
    for i in np.flatnonzero(labels == 0):
        hazard[i + 1:i + 4] = 0.05
    exceed = rng.random(n_seasons * n_days) < hazard
    r = fire_link.lead_ratio(labels, exceed, season, 0, 3, n_boot=300)
    check("suppressive regime gives a forward ratio below 1",
          r.ratio < 1 and r.ci_high < 1,
          f"ratio {r.ratio:.2f} [{r.ci_low:.2f},{r.ci_high:.2f}]")

    # A regime with no forward effect must not manufacture one.
    exceed_null = rng.random(n_seasons * n_days) < 0.20
    r0 = fire_link.lead_ratio(labels, exceed_null, season, 0, 3, n_boot=300)
    check("null regime interval covers 1",
          r0.ci_low < 1 < r0.ci_high,
          f"ratio {r0.ratio:.2f} [{r0.ci_low:.2f},{r0.ci_high:.2f}]")

    # Block bootstrap must stay wider than the naive independent-days one, for
    # the same reason as everywhere else in this pipeline.
    wide = fire_link.lead_ratio(labels, exceed, season, 0, 3, block_len=7,
                                n_boot=400)
    narrow = fire_link.lead_ratio(labels, exceed, season, 0, 3, block_len=1,
                                  n_boot=400)
    check("block bootstrap is wider than independent-days",
          (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low),
          f"block {wide.ci_high - wide.ci_low:.3f} "
          f"vs naive {narrow.ci_high - narrow.ci_low:.3f}")

    # Stratification: a regime that is nothing but a proxy for the covariate
    # must lose its apparent effect once the covariate is held fixed. This is
    # the test that keeps the FWI-increment claim honest.
    # The covariate has to persist, or an origin-day value cannot predict a
    # forward window at all and the test is vacuous - which is also why the
    # real version stratifies on FWI, a strongly autocorrelated field.
    cov = np.zeros(n_seasons * n_days)
    for i in range(1, len(cov)):
        cov[i] = 0.9 * cov[i - 1] + 0.4 * rng.standard_normal()
    hot = cov > np.quantile(cov, 0.7)
    proxy = hot.astype(int)                  # regime 1 <=> high covariate
    exceed_cov = rng.random(n_seasons * n_days) < np.where(hot, 0.5, 0.1)
    unstrat = fire_link.lead_ratio(proxy, exceed_cov, season, 1, 3, n_boot=300)
    strat = fire_link.lead_ratio_by_stratum(proxy, exceed_cov, cov, season, 1,
                                            3, n_strata=4, n_boot=300)
    inner = [s for s in strat if s.n_in >= 15 and np.isfinite(s.ratio)]
    check("pure covariate proxy shows an effect before stratifying",
          unstrat.ratio > 1.5, f"ratio {unstrat.ratio:.2f}")
    check("... and loses most of it within covariate strata",
          all(s.ratio < unstrat.ratio for s in inner),
          f"strata {[round(s.ratio, 2) for s in inner]} vs {unstrat.ratio:.2f}")

    check("strata partition every origin day",
          sum(s.n_in for s in strat) == int((proxy[
              fire_link.forward_window(exceed_cov, season, 3)[0]] == 1).sum()))


def test_composites():
    """The numerics behind the composite maps, without importing matplotlib.

    A composite is the one output here that is judged by eye, which makes a
    quiet error in it unusually dangerous: a map with the weighting left in,
    or with every point significant because days were counted as independent,
    still looks exactly like weather.
    """
    print("\ncomposites")
    from regimes_pt import plots

    # Three 5-day episodes of regime 0 among 100 days: 15 days, 3 independent
    # samples, not 15. Counting days would inflate any t-statistic by sqrt(5).
    labels = np.full(100, 1)
    for start in (10, 40, 70):
        labels[start:start + 5] = 0
    check("effective n counts episodes, not days",
          np.isclose(plots.effective_n(labels, 0), 3.0),
          f"{plots.effective_n(labels, 0):.1f} from 15 days in 3 episodes")

    # A regime with a known constant offset must come back with that offset.
    rng = np.random.default_rng(11)
    anom = rng.standard_normal((100, 6)) * 2.0
    anom[labels == 0] += 50.0
    mean, t, n_eff = plots.composite(anom, labels, 0)
    check("composite recovers the imposed anomaly",
          np.allclose(mean, 50.0, atol=1.5), f"mean {mean.mean():.1f} gpm")
    check("composite t uses the episode count",
          np.allclose(np.abs(t), np.abs(mean) / (anom[labels == 0].std(
              axis=0, ddof=1) / np.sqrt(3.0)), rtol=1e-6))

    # The weighting must never reach a map: unweight then composite has to
    # equal composite then unweight, and the pipeline does the former.
    area_w = np.sqrt(np.cos(np.deg2rad(np.linspace(30, 55, 6))))
    m_unw = plots.composite(anom / area_w, labels, 0)[0]
    check("unweighting commutes with compositing",
          np.allclose(m_unw, mean / area_w))

    check("an empty regime raises rather than returning NaNs",
          _raises(plots.composite, anom, labels, 3))


def test_cwt():
    """Jenkinson-Collison types against fields whose answer is known a priori.

    Every failure mode here is silent: a sign slip in the vorticity turns every
    cyclone into an anticyclone, a transposed grid point rotates the whole
    compass, and both produce a plausible-looking type distribution.
    """
    print("\ncwt")
    from regimes_pt import cwt

    lat = np.arange(20.0, 61.0)
    lon = np.arange(-40.0, 11.0)
    LO, LA = np.meshgrid(lon, lat)

    def types(fields):
        return cwt.classify(np.asarray(fields), lat, lon)

    # Geostrophic flow: in the NH, high pressure to the south gives a westerly.
    west = types([-LA])[0]
    east = types([LA])[0]
    south = types([LO])[0]
    north = types([-LO])[0]
    check("high to the south gives westerly", west == "W", west)
    check("high to the north gives easterly", east == "E", east)
    check("high to the east gives southerly", south == "S", south)
    check("high to the west gives northerly", north == "N", north)

    # A pure rotation: a bowl (low centre) must be cyclonic, a dome anticyclonic.
    r2 = (LO - cwt.CENTRE_LON) ** 2 + 4 * (LA - cwt.CENTRE_LAT) ** 2
    check("closed low classifies cyclonic", types([r2])[0] == "C", types([r2])[0])
    check("closed high classifies anticyclonic", types([-r2])[0] == "A",
          types([-r2])[0])

    # The three regimes of the rule are set by |Z|/F alone, so walk a westerly
    # flow of growing strength past a fixed low and check the type crosses the
    # boundaries in the right order: rotational, hybrid, then directional.
    walk = [types([-LA * a + r2 * 0.6])[0] for a in (8, 60, 200)]
    check("rule crosses |Z|/F = 2 and 1 in order",
          walk == ["C", "CW", "W"], " -> ".join(walk))

    hyb = walk[1]
    check("hybrid names both rotation and direction",
          len(hyb) > 1 and hyb[0] in "AC" and hyb[1:] in cwt.DIRECTIONS, hyb)

    p_h = cwt.sample(np.asarray([-LA * 60 + r2 * 0.6]), lat, lon)
    ratio = abs(cwt.indices(p_h)[2][0]) / cwt.indices(p_h)[3][0]
    check("hybrid really sits between F and 2F", 1.0 <= ratio <= 2.0,
          f"|Z|/F = {ratio:.2f}")

    # Scale invariance is what licenses running this on gpm instead of hPa.
    a = types([-LA * 3 + r2 * 0.4])[0]
    b = types([(-LA * 3 + r2 * 0.4) * 137.0])[0]
    check("classification is invariant to field scaling", a == b, f"{a} vs {b}")

    # Vorticity sign convention: cyclonic is positive.
    p = cwt.sample(np.asarray([r2]), lat, lon)
    check("cyclonic vorticity is positive", cwt.indices(p)[2][0] > 0)

    # The stencil must be the published one: 16 points, 10 by 5 degrees,
    # spanning 25W-5E and 30-50N when centred over Portugal.
    pts = cwt.grid_points()
    lons = sorted({x for x, _ in pts}); lats = sorted({y for _, y in pts})
    check("sixteen grid points", len(pts) == 16, str(len(pts)))
    check("grid spans 25W-5E, 30-50N",
          lons == [-25, -15, -5, 5] and lats == [30, 35, 40, 45, 50],
          f"{lons} {lats}")

    codes, vocab = cwt.to_codes(np.array(["A", "NE", "A", "C"], dtype=object))
    check("codes round-trip through the vocabulary",
          [vocab[c] for c in codes] == ["A", "NE", "A", "C"])


def test_cost_loss():
    """The decision model, against cases where value is known analytically.

    A cost-loss curve is easy to get wrong in a way that flatters the forecast:
    the reference is the *better* of two climatological strategies, and using
    the wrong one turns a worthless forecast into a valuable-looking one.
    """
    print("\ncost-loss")

    rng = np.random.default_rng(7)
    n = 4000
    event = rng.random(n) < 0.25

    # A perfect forecast scores 1 at every cost-loss ratio.
    perfect = event.astype(float)
    vals = [fire_link.cost_loss_value(perfect, event, a)
            for a in (.05, .25, .5, .8)]
    check("perfect forecast scores 1 everywhere",
          np.allclose(vals, 1.0), f"{np.round(vals, 4)}")

    # A constant forecast is one of the two climatological strategies, so it
    # can never beat the reference - value must be <= 0, never positive.
    const = np.full(n, 0.25)
    vals = [fire_link.cost_loss_value(const, event, a)
            for a in (.05, .2, .25, .3, .8)]
    check("constant forecast never scores above 0",
          max(vals) <= 1e-9, f"max {max(vals):.4f}")

    # Climatological value is 0 exactly at alpha = base rate, where the two
    # climatological strategies cost the same.
    check("value is 0 at alpha equal to the base rate",
          abs(fire_link.cost_loss_value(const, event, 0.25)) < 1e-9)

    # A forecast anti-correlated with the truth must be actively harmful.
    check("inverted forecast has negative value",
          fire_link.cost_loss_value(1.0 - perfect, event, 0.25) < 0)

    # An informative but imperfect forecast lands strictly between the two.
    noisy = np.clip(0.5 * event + rng.normal(0, .15, n) + .2, 0, 1)
    v = fire_link.cost_loss_value(noisy, event, 0.25)
    check("imperfect forecast lands strictly between 0 and 1",
          0 < v < 1, f"{v:.3f}")

    # Leave-one-year-out probabilities must not leak the scored year.
    years = np.repeat(np.arange(20), 200)
    # Give one year a wildly different rate; its fitted probability must come
    # from the other years, so it cannot match its own rate.
    ev = rng.random(n) < 0.2
    ev[years == 3] = True
    cells = np.zeros(n, dtype=int)
    p = fire_link.loyo_probabilities(cells, ev, years, 1)
    check("leave-one-year-out does not fit the scored year",
          p[years == 3].mean() < 0.5,
          f"year 3 is all-event but scored at p={p[years == 3].mean():.3f}")

    check("curve returns one value per alpha",
          len(fire_link.cost_loss_curve(noisy, event, [.1, .2, .3])) == 3)


def main() -> int:
    print("=" * 60)
    print("regimes_pt unit tests")
    print("=" * 60)
    test_preprocess()
    test_cluster()
    test_fire_link()
    test_regrid()
    test_download_plan()
    test_multifile()
    test_fire_layer()
    test_forecast_penalty()
    test_free_lead()
    test_composites()
    test_cwt()
    test_cost_loss()
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
        return 1
    print("all unit tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
