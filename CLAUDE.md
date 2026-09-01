# CLAUDE.md

Context for Claude Code working in this repository.

## What this is

A prototype that tests one scientific question:

> Does a Portugal-tuned weather-regime classification separate the Iberian
> extreme-fire circulation (upper ridge over NW Africa + cut-off low west of
> Portugal) better than the canonical Euro-Atlantic four-regime partition?

**The circulation in that question is half wrong**, as of the composite maps.
The cut-off low is real — regime 2 is the only one of the four with a closed
low in the composite total height field, at 42°N 14°W. The NW Africa ridge is
not: that box averages −25.9 gpm, the wrong sign. The ridge is at 55°N 14°W,
due north of the low, which is a blocking ridge with a cut-off low beneath it
— the textbook way cut-off lows form. Do not go looking for the subtropical
ridge; it was design intent, never a finding. Carmo et al. (2022) reach the
same picture independently — fire-driving anticyclones "between the Azores and
the British Isles", plus a rarer "inverse omega" with a closed low off the
Portuguese west coast — and never place the ridge over NW Africa either.

It is **not** a fire-probability product. The defensible output is a
regime-conditioned **odds ratio** — "under regime X, Beira Interior has 2.8×
baseline odds of a p95 FWI day" — not "38% chance of fire".

Operational context: ICNF / GFR Algarve, prontidão planning — **3–5 day lead,
not 2–6 weeks**. The original sub-seasonal framing did not survive contact
with the data and is kept here only because it explains older design choices.
`--step forecast` showed the signal needs 68–76% categorical regime accuracy,
which summer sub-seasonal forecasting does not have; `--step lead` showed the
same signal carries 3–5 usable days from the *analysed* regime, at no skill
cost. Still not tactical forecasting: the unit is a multi-day readiness
posture, not a dispatch decision.

## Current state

| | status |
|---|---|
| Pipeline code | written, unit-tested (53 assertions) |
| Synthetic end-to-end validation | passing, 7/7 |
| Real ERA5 + CEMS run | done — 1980–2025 JJAS, 5612 days |
| Fire layer (ICNF, ≥100 ha) | done — `--step fires` |
| Forecast penalty | done — `--step forecast` |
| Free lead + FWI increment | done — `--step lead` |
| Composite Z500 maps | done — `--step maps`; they falsified half the domain rationale |
| Selection-bias treatment | **not done** — regimes were chosen by max odds ratio |

The headline result: regime 0 halves the severe-fire-day rate over the
following five days, and it survives stratification by FWI (0.49 in the top
FWI quartile, 0.154 vs 0.316), so it adds to fire danger rather than
restating it. Regimes govern **escape, not ignition** — testing against an
absolute FWI threshold instead finds nothing, which is how this layer was
once mistakenly written off.

## Layout

```
regimes_pt/config.py       domains, seasons, preprocessing defaults
regimes_pt/download.py     CDS retrieval — ERA5 Z500, CEMS FWI reanalysis
regimes_pt/preprocess.py   harmonic climatology, detrend, area weight, EOF
regimes_pt/cluster.py      k-means, classifiability index, assignment, transitions
regimes_pt/fire_link.py    regionalisation, composites, odds ratios, CV skill
regimes_pt/plots.py        composite maps — optional, needs matplotlib + cartopy
run_prototype.py           CLI: download | regimes | compare | fires | forecast | lead | maps
tests/test_units.py        invariant tests, no network, ~10 s
tests/test_synthetic.py    ground-truth end-to-end, no network, ~3 min
docs/README.md             related work — read before claiming novelty
```

`docs/README.md` matters more than its size suggests. Carmo et al. (2022) is
the same question on the same country with a different method, and it
independently reaches the composite result in `--step maps`. It also names two
live problems: regime 2 may be blending two synoptic pathways that IPMA keeps
separate, and `canonical_summer` is a straw-man benchmark next to NE+E CWT
frequency. The papers themselves are untracked for licensing reasons.

## Commands

```bash
pip install -r requirements.txt
python tests/test_units.py                                       # always run first
python tests/test_synthetic.py
python run_prototype.py --step download
python run_prototype.py --step regimes --domain canonical_summer --k 4
python run_prototype.py --step regimes --domain pt_tuned --select-k
python run_prototype.py --step compare
python run_prototype.py --step fires                             # needs data/ocoPT_*.csv
python run_prototype.py --step forecast                          # how much skill it needs
python run_prototype.py --step lead                              # how much lead is free
python run_prototype.py --step maps                              # composites; needs cartopy
```

CDS credentials go in `~/.cdsapirc`, never in the repo.

## Invariants — do not "simplify" these

Each of these looks like an optimisation opportunity and is not. Breaking any of
them produces output that still looks like weather and is wrong.

1. **Detrend summer Z500.** Thermal expansion of the warming troposphere gives a
   large positive trend. Left in, k-means returns a cluster that is really
   "recent years", and regime frequencies show spurious trends. Default removes
   both the daily area-weighted domain mean and a per-gridpoint linear trend.

2. **Do not whiten the PCs.** Leading PCs carry more physical amplitude and must
   dominate the distance metric. `StandardScaler` before k-means gives a
   grid-scale noise EOF the same weight as the NAO. There is a unit test for this.

3. **Area-weight with sqrt(cos lat)** before the EOF step, so Euclidean distance
   in truncated space approximates area-weighted L2 on the sphere.

4. **Assign by spatial correlation, not Euclidean distance.** Regime membership
   is about the *shape* of the anomaly, not its amplitude. Days below the
   correlation threshold get label −1 and stay unclassified. A large unclassified
   fraction in summer is a real property of the atmosphere, not a bug to fix.

5. **Block bootstrap, never naive.** Regimes persist ~a week; FWI is strongly
   autocorrelated. Independent-days intervals inflate effective sample size by
   roughly an order of magnitude. `block_len` should track regime persistence.

6. **Leave-one-year-out for skill.** Never fit exceedance rates on the same year
   you score. Split by year, not by day — adjacent days are near-duplicates.

7. **Select k by margin over the null, not bare significance.** On real data
   several k are usually significant at once.

8. **Reuse training climatology when assigning new days.** `project_and_assign`
   takes the climatology and weights as arguments for this reason. Recomputing
   them on a short forecast window is silent leakage.

9. **Never let a window span two seasons.** Consecutive rows of a JJAS archive
   put 30 September beside 1 June of the next year. A rolling mean across that
   gap does not raise — it quietly averages September fire days into a June
   forecast and inflates the apparent lead. `block_aggregate` and
   `forward_window` both cut within seasons and drop the ragged tail.

10. **Stratify by FWI before claiming the regime adds anything.** ICNF already
    runs on fire danger. An unstratified regime effect can be pure repackaging
    of FWI, and `lead_ratio_by_stratum` is the test that separates the two —
    there is a unit test where a regime that *is* a covariate proxy collapses
    from 2.62 to ~1.0 under stratification. Stratify on the window mean, not
    the origin day, or the comparison understates what an FWI forecast knows.

## Known gotchas

- **CDS dataset identifiers drift.** If a request 404s, check the live catalogue
  and update the constants in `download.py`. Do not assume the code is broken.
- **ERA5 time coordinate is sometimes `valid_time`**, sometimes `time`, depending
  on dataset and vintage. `open_z500` handles both; new readers must too.
- **Geopotential is m²/s², not metres.** `open_z500` divides by g. Composite maps
  should be in gpm.
- **CEMS FWI masks ocean as NaN.** Drop all-NaN columns before regionalising or
  k-means will fail or silently cluster the mask.
- **Z500 and FWI calendars must be intersected**, not assumed aligned — different
  products, different leap-day and time-zone handling. `step_compare` does this.
- **Memory.** 46 JJAS seasons of 1° Z500 over the canonical domain is fine in
  RAM; 0.25° is not. Do not raise the grid resolution without a chunking plan.
  Regime patterns are planetary scale — 1° is ample.

## Conventions

- Plain numpy in the numerical core so it stays testable without network or files;
  xarray only at the I/O boundary (`prepare_from_dataarray`, `open_z500`).
- Every new statistical claim gets a unit test pinning the property, not just the
  return type.
- Docstrings explain *why* a choice was made where the choice is non-obvious.
  Keep that — it is the difference between a defensible method and a black box.
- Comments and identifiers in English; user-facing operational output may be
  Portuguese.

## What is deliberately not built

- **Layer 2, regime → burned area.** Needs the ICNF fire database. Expect it to be
  much weaker: most variance in Portuguese fire outcomes is ignition and
  suppression, not meteorology. Keep it visibly separate so ignition noise does
  not contaminate the S2S signal.
- **Live forecast path.** Hindcast only. Would need licensed ECMWF regime
  probabilities, or a GEFS/IFS open-data Z500 pull projected onto fitted
  centroids via `cluster.project_and_assign`.
- **The CWT benchmark.** `--step compare` scores `pt_tuned` against
  `canonical_summer`, a partition built for the Euro-Atlantic winter. The
  benchmark that would actually settle anything is NE+E circulation-weather-type
  frequency (Trigo & DaCâmara 2000, as used by Carmo et al. 2022): cheap,
  validated for this region, and what IPMA would reach for. See `docs/README.md`.

- **Regionalised composites.** `--step maps` draws the national picture only.
  Per-region composites would test whether the North/Centre signal and the
  (much thinner) South signal come from the same circulation.
