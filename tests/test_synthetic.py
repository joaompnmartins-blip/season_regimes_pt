"""End-to-end validation on synthetic data with known ground truth.

ERA5 and CEMS are not reachable from a sandbox, so this script substitutes a
synthetic Z500-like archive in which the answer is known by construction: five
quasi-stationary patterns, visited by a persistent Markov chain, buried under a
seasonal cycle, a warming trend and red noise of realistic amplitude.

If the pipeline is correct it must:
  1. select k = 5 as significant against the red-noise null;
  2. recover all five implanted patterns with high spatial correlation;
  3. identify the one pattern that drives the synthetic fire weather;
  4. return an odds ratio for that regime whose bootstrap CI excludes 1;
  5. produce positive out-of-sample Brier skill;
  6. show block-bootstrap intervals materially wider than naive ones.

Passing this proves the machinery. It says nothing about the atmosphere - that
requires the real ERA5 + CEMS run.
"""

from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from regimes_pt import cluster, fire_link, preprocess  # noqa: E402


TRUE_K = 5
LAT = np.arange(55.0, 24.9, -2.5)
LON = np.arange(-40.0, 10.1, 2.5)
YEARS = np.arange(1980, 2026)
DOY_RANGE = np.arange(152, 274)          # 1 Jun - 30 Sep


def build_patterns() -> np.ndarray:
    """Five smooth, distinct large-scale patterns on the domain."""
    lo, la = np.meshgrid(np.deg2rad(LON), np.deg2rad(LAT))
    P = np.stack([
        # 0: zonal dipole (NAO-like)
        np.sin(2.2 * la) * np.cos(1.0 * lo),
        # 1: reversed dipole
        -np.sin(2.2 * la) * np.cos(1.0 * lo),
        # 2: mid-Atlantic ridge
        np.exp(-(((lo + 0.45) / 0.35) ** 2 + ((la - 0.72) / 0.22) ** 2)),
        # 3: northern blocking high
        np.exp(-(((lo - 0.05) / 0.40) ** 2 + ((la - 0.92) / 0.18) ** 2)),
        # 4: THE FIRE PATTERN - subtropical ridge to the southeast with a
        #    cut-off low to the west of the domain centre
        (np.exp(-(((lo - 0.10) / 0.30) ** 2 + ((la - 0.60) / 0.16) ** 2))
         - 0.9 * np.exp(-(((lo + 0.35) / 0.22) ** 2 + ((la - 0.70) / 0.14) ** 2))),
    ])
    P -= P.mean(axis=(1, 2), keepdims=True)
    P /= P.std(axis=(1, 2), keepdims=True)
    return P


def markov_sequence(n: int, k: int, self_p: float, rng) -> np.ndarray:
    """Persistent Markov chain -> realistic regime episode lengths."""
    seq = np.empty(n, dtype=int)
    seq[0] = rng.integers(k)
    for i in range(1, n):
        if rng.random() < self_p:
            seq[i] = seq[i - 1]
        else:
            choices = [c for c in range(k) if c != seq[i - 1]]
            seq[i] = rng.choice(choices)
    return seq


def build_dataset(rng):
    doy = np.tile(DOY_RANGE, len(YEARS))
    years = np.repeat(YEARS, len(DOY_RANGE))
    n = len(doy)

    patterns = build_patterns()
    truth = markov_sequence(n, TRUE_K, self_p=0.86, rng=rng)

    nlat, nlon = len(LAT), len(LON)
    field = np.zeros((n, nlat, nlon))

    # Regime signal, with day-to-day amplitude variability.
    amp = 55.0 * (0.6 + 0.8 * rng.random(n))
    for i in range(n):
        field[i] += amp[i] * patterns[truth[i]]

    # Red noise at a realistic amplitude relative to the signal.
    noise = np.zeros((n, nlat, nlon))
    e = rng.normal(0, 38.0, size=(n, nlat, nlon))
    smooth = np.ones((3, 3)) / 9.0
    for i in range(1, n):
        noise[i] = 0.72 * noise[i - 1] + e[i]
    # crude spatial smoothing so the noise is not pure grid-scale
    from scipy.signal import convolve2d
    for i in range(n):
        noise[i] = convolve2d(noise[i], smooth, mode="same", boundary="symm")
    field += noise

    # Seasonal cycle and a warming trend (both must be removed by preprocessing).
    seas = 30.0 * np.cos(2 * np.pi * (doy - 200) / 365.25)
    field += seas[:, None, None]
    field += (0.9 * (years - years.mean()))[:, None, None]
    field += 5700.0

    return field, doy, years, truth, patterns


def build_fwi(truth, doy, rng):
    """Synthetic FWI over three 'regions' with different regime sensitivity.

    Region 0 responds strongly to the fire pattern (regime 4), region 1 weakly,
    region 2 not at all. The pipeline should recover exactly that ordering.
    """
    n = len(truth)
    seas = 12.0 * np.exp(-((doy - 215) / 45.0) ** 2)
    sens = np.array([14.0, 6.0, 0.0])

    series = []
    for s in sens:
        base = np.zeros(n)
        e = rng.normal(0, 6.0, n)
        for i in range(1, n):
            base[i] = 0.80 * base[i - 1] + e[i]
        series.append(20.0 + seas + base + s * (truth == 4))
    return np.column_stack(series)


def main() -> int:
    rng = np.random.default_rng(20260817)
    print("Building synthetic archive ...")
    field, doy, years, truth, patterns = build_dataset(rng)
    print(f"  field {field.shape}, {len(YEARS)} years, {len(DOY_RANGE)} days/season")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prep = preprocess.prepare(
            field, LAT, LON, times=np.arange(len(doy)), doy=doy,
            detrend="both", var_frac=0.90, max_eofs=30,
        )
    print(f"  retained {prep.eof.n_eof} EOFs "
          f"({100*prep.eof.explained.sum():.1f}% of variance)")

    print("\nClassifiability index vs red-noise null "
          "(reduced settings for speed):")
    sel = cluster.select_k(
        prep.eof.pcs, k_range=(2, 3, 4, 5, 6, 7, 8),
        n_partitions=30, n_surrogates=20, n_init=8,
        n_partitions_surrogate=12, random_state=1,
    )
    sig = [s.k for s in sel if s.significant]
    best = max(sel, key=lambda s: s.ci - s.ci_null_p95)
    print(f"  significant k: {sig}")
    print(f"  largest margin over null at k={best.k}")

    check1 = TRUE_K in sig
    print(f"  [{'PASS' if check1 else 'FAIL'}] true k={TRUE_K} flagged significant")

    # --- pattern recovery -------------------------------------------------
    labels, cent_pc = cluster.fit_kmeans(prep.eof.pcs, TRUE_K, n_init=200,
                                         random_state=7)
    patterns_w = cluster.centroids_to_patterns(cent_pc, prep.eof.eofs)

    true_flat_w = (patterns.reshape(TRUE_K, -1) * prep.area_w)
    C = cluster._pattern_corr_matrix(patterns_w, true_flat_w)
    matched = C.max(axis=1)
    print("\nPattern recovery (correlation with best-matching implanted pattern):")
    for i, (m, j) in enumerate(zip(matched, C.argmax(axis=1))):
        print(f"  centroid {i} -> true pattern {j}   r = {m:.3f}")
    check2 = matched.min() > 0.80 and len(set(C.argmax(axis=1))) == TRUE_K
    print(f"  [{'PASS' if check2 else 'FAIL'}] all five patterns recovered, min r > 0.80")

    fire_centroid = int(np.argmax(C[:, 4]))
    assign = cluster.assign_days(prep.anom_w, patterns_w, corr_threshold=0.40)
    print(f"\n  unclassified fraction: {assign.unclassified_frac:.1%}")
    print(f"  mean persistence (days): "
          f"{np.round(cluster.mean_persistence(assign.labels, TRUE_K), 1)}")

    # --- fire link --------------------------------------------------------
    fwi = build_fwi(truth, doy, rng)
    print("\nRegime-conditioned fire-weather diagnostics (p95 exceedance):")
    names = ["region 0 (strong)", "region 1 (weak)", "region 2 (null)"]
    checks = []
    for r in range(3):
        exc = fire_link.exceedance(fwi[:, r], 95.0, doy=doy, window=15)
        comp = fire_link.fire_day_composite(prep.anom_w, exc)
        ms = fire_link.match_scores(comp, patterns_w)
        odds = fire_link.odds_by_regime(labels, exc, TRUE_K, block_len=7,
                                        n_boot=600, random_state=3)
        skill = fire_link.cv_brier_skill(labels, exc, years, TRUE_K)
        o = odds[fire_centroid]
        print(f"\n  {names[r]}")
        print(f"    best composite match: centroid {int(np.argmax(ms))} "
              f"r={ms.max():.3f}  (fire centroid r={ms[fire_centroid]:+.3f})")
        print(f"    OR for fire regime: {o.odds_ratio:.2f} "
              f"[{o.ci_low:.2f}, {o.ci_high:.2f}]")
        print(f"    cross-validated BSS: {skill.bss:+.4f}   AUC {skill.auc:.3f}")
        checks.append((r, o, skill, int(np.argmax(ms))))

    r0, o0, s0, m0 = checks[0]
    check3 = m0 == fire_centroid
    check4 = o0.ci_low > 1.0
    check5 = s0.bss > 0.0
    check6 = checks[2][1].ci_low <= 1.0 <= checks[2][1].ci_high
    print(f"\n  [{'PASS' if check3 else 'FAIL'}] composite for region 0 matches the fire regime")
    print(f"  [{'PASS' if check4 else 'FAIL'}] odds-ratio CI excludes 1 for region 0")
    print(f"  [{'PASS' if check5 else 'FAIL'}] positive out-of-sample BSS for region 0")
    print(f"  [{'PASS' if check6 else 'FAIL'}] null region shows no significant effect")

    # --- bootstrap sanity -------------------------------------------------
    exc0 = fire_link.exceedance(fwi[:, 0], 95.0, doy=doy, window=15)
    naive = fire_link.odds_by_regime(labels, exc0, TRUE_K, block_len=1,
                                     n_boot=600, random_state=4)[fire_centroid]
    w_block = o0.ci_high - o0.ci_low
    w_naive = naive.ci_high - naive.ci_low
    check7 = w_block > 1.25 * w_naive
    print(f"\n  CI width: block(7d) {w_block:.2f} vs naive(1d) {w_naive:.2f} "
          f"-> ratio {w_block/w_naive:.2f}")
    print(f"  [{'PASS' if check7 else 'FAIL'}] block bootstrap is materially wider")

    allc = [check1, check2, check3, check4, check5, check6, check7]
    print(f"\n{'=' * 55}\n{sum(allc)}/{len(allc)} checks passed\n{'=' * 55}")
    return 0 if all(allc) else 1


if __name__ == "__main__":
    raise SystemExit(main())
