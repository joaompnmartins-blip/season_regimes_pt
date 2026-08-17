# TESTING.md — staged validation plan

Work through these in order. Each stage has a command, an acceptance criterion,
and a **what failure means** note. Do not proceed past a failing stage — later
stages will produce plausible output on a broken pipeline, which is worse than
an error.

Stages 0–2 need no network. Stage 3 onward needs a CDS account.

---

## Stage 0 — environment

```bash
pip install -r requirements.txt
python -c "import numpy, scipy, sklearn, pandas, xarray, netCDF4; print('ok')"
python tests/test_units.py
```

**Accept:** all 39 unit assertions pass, runtime under ~15 s.

**If it fails:** a numerical invariant is broken. Read which assertion failed —
each pins one specific property (see `CLAUDE.md` §Invariants). Do not proceed.

---

## Stage 1 — synthetic ground truth

```bash
python tests/test_synthetic.py
```

**Accept:** 7/7 checks pass. Expected values from the reference run:

| check | expected |
|---|---|
| true k=5 flagged significant | yes, largest margin at k=5 |
| pattern recovery | r = 0.987–0.993, all 5 distinct |
| mean persistence | 6.8–7.8 days |
| zone 0 odds ratio | ~14.9, CI excludes 1 |
| zone 0 cross-validated BSS | ~+0.095 |
| zone 2 (null zone) | CI spans 1 |
| block vs naive CI width | ratio ~1.5 |

**Known artefact:** the classifiability index saturates near 1.000 for nearly
every k here, because the synthetic signal is far cleaner and lower-rank than
reality (only 4 EOFs reach 90% variance). The index has little resolving power in
this regime; the *margin* criterion still picks k=5. This is expected on synthetic
data and is a **red flag on real data** — see Stage 5.

---

## Stage 2 — extend the unit suite

Tasks for Claude Code, before touching real data. Add to `tests/test_units.py`,
keeping the `check()` reporting style.

1. **Detrend efficacy under a realistic trend.** Build a synthetic field with a
   0.9 gpm/yr uniform trend plus regime structure. Assert that after
   `prepare(detrend="both")`, regression of each regime's annual frequency on
   year has |slope| below noise, and that with `detrend="none"` at least one
   regime shows a significant trend. This is the single most important guard in
   the pipeline.
2. **Leakage guard.** Assert `cv_brier_skill` returns BSS ≤ ~0 when labels are
   randomly permuted *within* year but exceedance is held fixed.
3. **Calendar alignment.** Feed `step_compare`-style inputs with deliberately
   mismatched Z500 and FWI date ranges (one leap year offset) and assert the
   intersection length is correct and no silent index shift occurs.
4. **`project_and_assign` leakage.** Assert that assigning the training days via
   `project_and_assign` with the training climatology reproduces `assign_days`
   labels exactly, and that recomputing climatology on a 30-day window changes
   them — demonstrating why the argument exists.
5. **Threshold sensitivity.** Sweep `corr_threshold` over 0.2–0.6 and record the
   unclassified fraction; assert monotonicity.

**Accept:** all new tests pass and test 1 fails when detrending is disabled.

---

## Stage 3 — download smoke test

Before queueing 46 years, pull one season.

```bash
python run_prototype.py --step download --data-dir ./data_smoke  # then Ctrl-C after year 1
python - <<'PY'
from regimes_pt import download
import glob
da = download.open_z500(sorted(glob.glob("data_smoke/z500_pt_tuned_JJAS_1980.nc")))
print(da.dims, da.shape)
print("gpm range:", float(da.min()), float(da.max()))
PY
```

**Accept:** dims `(time, latitude, longitude)`, 122 time steps for JJAS,
values roughly **5500–6000 gpm**.

**If values are ~5e4:** the m²/s² → gpm division did not happen. **If ~122×4
time steps:** the hourly fallback was used and sub-daily averaging failed.
**If a 404:** the CDS identifier has changed — check the catalogue, update
`download.py`, do not work around it.

---

## Stage 4 — full download

```bash
python run_prototype.py --step download
```

Hours in the CDS queue; resumable (existing files are skipped). Verify the count:
46 years × 2 domains + 46 FWI files.

---

## Stage 5 — regime fit, canonical baseline

```bash
python run_prototype.py --step regimes --domain canonical_summer --k 4
```

**Accept:**

| quantity | expected on real ERA5 JJAS |
|---|---|
| retained EOFs at 90% variance | **15–25** |
| regime frequencies | roughly 15–35% each |
| mean persistence | 5–10 days |
| unclassified fraction | 25–45% at threshold 0.40 |

Then sanity-check the physics: the four centroids should be recognisable as
Atlantic Low, Blocking, Atlantic Ridge and a summer NAO−. If they are not, the
preprocessing is wrong, not the literature.

**Red flags:**

| symptom | likely cause |
|---|---|
| retained EOFs < 8 | under-truncation; variance collapsed, check detrending |
| classifiability index ≈ 1.000 for all k | same — problem is too low-dimensional |
| any regime frequency < 8% | degenerate cluster; k too high |
| mean persistence < 3 days | not regimes, noise |
| unclassified > 60% | anomalies wrong, or threshold too strict |
| a regime's annual frequency trends monotonically | **detrending failed** |

That last one deserves an explicit check — plot annual regime frequency against
year and regress. A monotonic trend means the warming signal survived and every
downstream number is contaminated.

---

## Stage 6 — Portugal-tuned fit

```bash
python run_prototype.py --step regimes --domain pt_tuned --select-k
```

Inspect `out/ksel_pt_tuned.json`. Expect CI in the **0.7–0.9** range with a
visible peak, not saturation.

**Accept:** at least one k significant against the red-noise null; the
margin-over-null criterion picks a k in the 4–8 range.

Fit the selected k, then also fit k=4 for a like-for-like comparison against the
canonical baseline (domain effect isolated from k effect). This matters — if the
tuned domain wins at k=7 you have not shown the domain helped, only that more
clusters help.

---

## Stage 7 — the actual test

```bash
python run_prototype.py --step compare
```

**The primary result** is the composite-match score per fire-climate zone: build
the mean Z500 anomaly over each zone's p95 FWI days, correlate against every
centroid, take the best.

| best match | interpretation |
|---|---|
| < 0.4 | no regime captures the fire-day circulation; guidance will be weak regardless of presentation |
| 0.4–0.6 | partial; usable as a background-odds modulator only |
| > 0.6 | the partition genuinely resolves the fire pattern |

**The hypothesis is confirmed** if canonical k=4 tops out below ~0.5 for the
Interior zones while the tuned partition clears it comfortably, *and* the tuned
configuration wins on cross-validated BSS. That combination is a publishable
result, not just a working tool.

**The hypothesis is refuted** if both score similarly. That is a real and useful
finding — report it. Do not tune the domain until it wins.

**Expected magnitudes.** BSS in the **0.02–0.15** range is realistic and useful.
BSS > 0.3 is suspicious — check for leakage before celebrating. Odds ratios of
2–4× for the fire-favouring regime are plausible; 15× as in the synthetic test is
not, because real regimes explain far less variance than implanted ones.

---

## Stage 8 — robustness sweeps

The result must survive its own arbitrary choices. Vary one at a time and record
whether the Stage 7 conclusion flips:

- `corr_threshold` ∈ {0.3, 0.4, 0.5}
- `var_frac` ∈ {0.85, 0.90, 0.95}
- exceedance percentile ∈ {90, 95, 98}
- season ∈ {JJA, JJAS, MJJASO} — MJJASO matters for October events
- `n_regions` ∈ {4, 6, 8}
- `detrend` ∈ {"domain_mean", "both"}
- `block_len` ∈ {5, 7, 10}

**Accept:** the sign and rough magnitude of the canonical-vs-tuned difference is
stable across the sweep. If the conclusion flips on a threshold change, you have
a threshold artefact, not a result — say so.

Raise `n_surrogates` to 500 for the final classifiability numbers.

---

## Stage 9 — reporting

Produce `out/report.md` containing: k-selection curves for both domains,
composite-match table by zone and configuration, odds ratios with block-bootstrap
CIs, the BSS comparison table, and the robustness sweep.

State the unclassified fraction prominently. The days that matter most
operationally — extreme fire weather — are exactly the days most likely to be
assigned −1, and a reader who does not know that will over-trust the product.

Frame every conditional statement as an odds ratio relative to climatology, never
as a probability of fire.
