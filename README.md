# regimes_pt — regime classifier prototype + Portugal discrimination test

Tests whether a Portugal-tuned weather-regime classification separates the
Iberian extreme-fire circulation better than the canonical Euro-Atlantic four.

## Status

The **code is validated**; the **science is not run yet**. ERA5 and CEMS are not
reachable from this sandbox, so the pipeline was validated end-to-end against a
synthetic archive with known ground truth. The real answer needs the CDS
download, which you run locally.

## Install and run

```bash
pip install numpy scipy scikit-learn pandas xarray netCDF4 cdsapi matplotlib

python tests/test_synthetic.py                                   # validate first
python run_prototype.py --step download                          # hours, once
python run_prototype.py --step regimes --domain canonical_summer --k 4
python run_prototype.py --step regimes --domain pt_tuned --select-k
python run_prototype.py --step compare
```

Needs a CDS account and `~/.cdsapirc`. CDS dataset identifiers drift — if a
request 404s, check the catalogue and update the constants in `download.py`.

## The comparison

| | canonical | pt_tuned |
|---|---|---|
| domain | 90°W–30°E, 20–80°N | 40°W–10°E, 25–55°N |
| k | 4 (imposed, literature) | selected by classifiability index |
| S2S compatibility | direct — ECMWF products project onto these | none — project forecast fields onto your own centroids |

Two diagnostics decide it:

**Composite match.** Build the mean Z500 anomaly over each zone's p95 FWI days
and correlate it against every regime centroid. If the best match is below about
0.5, no regime in that partition captures the fire-day circulation, and
regime-conditioned guidance for that zone will be weak regardless of how the
probabilities are presented.

**Cross-validated Brier skill.** Leave-one-year-out, forecasting
P(p95 FWI exceedance | today's regime). Deliberately trivial as a forecast, but
it is the right comparator: any BSS difference between configurations is
attributable to the regime definition alone.

## Method notes

**Detrending is not optional.** Summer Z500 has a large positive trend from
tropospheric thermal expansion. Left in, k-means returns a cluster that is really
"recent years" and regime frequencies show spurious trends. Default removes both
the daily area-weighted domain mean and a per-gridpoint linear trend.

**PCs are not whitened.** Leading PCs carry more physical amplitude and should
dominate the distance metric; standardising them gives a grid-scale noise EOF the
same weight as the NAO.

**k is chosen, not assumed.** `select_k` implements the Michelangeli et al.
(1995) classifiability index against an AR(1) red-noise null that preserves each
PC's variance and lag-1 autocorrelation. On real data expect several k to be
significant at once — use the *margin* over the null, not bare significance.

**Days may be unclassified.** Assignment is by spatial correlation with a
threshold (default 0.40), so weak-anomaly days get label −1. Expect a large
unclassified fraction in summer. That is a property of the atmosphere; forcing
every day into a regime is how these classifications get oversold.

**Block bootstrap everywhere.** Regimes persist ~a week and FWI is strongly
autocorrelated. Treating days as independent inflates the effective sample size
by roughly an order of magnitude. In the synthetic test the 7-day block interval
came out 1.5× wider than the naive one — on real data with stronger persistence
the gap will be larger.

**Regions come from the data.** `regionalize` clusters FWI grid points on
standardised time series, so it keys on co-variability rather than mean level.
Distritos cut straight across the Litoral/Interior gradient.

## Synthetic validation — 7/7 passed

Five implanted patterns, persistent Markov chain (self-transition 0.86), buried
under a seasonal cycle, warming trend and spatially-smoothed red noise. One
pattern drives synthetic FWI in zone 0 strongly, zone 1 weakly, zone 2 not at all.

```
true k=5 flagged significant; largest margin over null at k=5
all 5 patterns recovered, r = 0.987–0.993
mean persistence 6.8–7.8 days
zone 0 (strong): composite matches fire regime r=0.999, OR 14.9 [10.2, 22.8], BSS +0.095
zone 1 (weak):                                  r=0.965, OR  3.3 [ 2.2,  4.6], BSS +0.011
zone 2 (null):   OR 0.95 [0.55, 1.48] — correctly not significant
block CI 1.52× naive CI
```

**One caveat from the run itself.** The classifiability index saturated near
1.000 for almost every k, because the synthetic signal is far cleaner and
lower-rank than reality (only 4 EOFs reached 90% variance; ERA5 will need
15–25). The index had little resolving power there and flagged k=2,4,5,6,7,8 all
as significant — the *margin* criterion still picked k=5 correctly. On real data
CI values should land around 0.7–0.9 with a more discriminating peak. If they
come back near 1.0 again, something is wrong with the preprocessing — most
likely under-truncation.

## Limitations

- **Layer 2 is not built.** This is regime → fire *weather*. Regime → burned area
  needs the ICNF database and will be far weaker: most variance in Portuguese
  fire outcomes is ignition and suppression, not meteorology.
- **No live forecast path.** Hindcast only. Operational use needs either licensed
  ECMWF regime probabilities or a GEFS/IFS open-data Z500 pull projected onto the
  fitted centroids (`cluster.project_and_assign` is written for this — it reuses
  the training climatology and weights, which is where leakage usually creeps in).
- **Extremes are exactly where clustering is weakest.** The days that matter most
  operationally are the ones most likely to be assigned −1.
- **The right output is an odds ratio, not a fire probability.** "Under regime X,
  Beira Interior has 2.8× baseline odds of a p95 FWI day" is defensible;
  "38% chance of fire" is not.

## Layout

```
regimes_pt/config.py       domains, seasons, preprocessing defaults
regimes_pt/download.py     CDS retrieval for ERA5 Z500 and CEMS FWI
regimes_pt/preprocess.py   harmonic climatology, detrend, area weight, EOF
regimes_pt/cluster.py      k-means, classifiability index, assignment, transitions
regimes_pt/fire_link.py    regionalisation, composites, odds ratios, CV skill
run_prototype.py           three-step CLI
tests/test_synthetic.py    ground-truth validation, no network needed
```
