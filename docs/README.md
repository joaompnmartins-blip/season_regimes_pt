# Related work

Notes on the two papers this prototype should be read against. Both are
Portuguese, both use the same circulation-typing tradition, and one of them
independently reaches the conclusion our composite maps reached in
`--step maps`.

The PDFs and their text extracts are **not tracked** (see `.gitignore`):
Trigo and DaCâmara (2000) is all-rights-reserved, and Carmo et al. (2022) is
CC BY-NC-**ND**, which a text extraction arguably violates. Fetch them from the
DOIs below.

Read the **figures and tables**, not a text extraction. The extracts drop
Carmo's Table 1 (the full 26-type chi-square), Table 2 (Z500 over Portugal per
type), Figure 4 (the day-by-day precursor) and Trigo's Figure A1 (where each
type's pressure centre actually sits). Every quantitative statement below came
from those, and one earlier claim in this file was wrong until they were read.

- Trigo, R.M. & DaCâmara, C.C. (2000). Circulation weather types and their
  influence on the precipitation regime in Portugal. *International Journal of
  Climatology* 20, 1559–1581.
- Carmo, M., Ferreira, J., Mendes, M., Silva, Á., Silva, P., Alves, D., Reis, L.,
  Novo, I. & Viegas, D.X. (2022). The climatology of extreme wildfires in
  Portugal, 1980–2018: contributions to forecasting and preparedness.
  *International Journal of Climatology* 42(5), 3123–3146.
  [doi:10.1002/joc.7411](https://doi.org/10.1002/joc.7411)

---

## Trigo & DaCâmara (2000) — the classification everyone else uses

Defines the **Circulation Weather Type** (CWT) scheme for Portugal: daily
geostrophic flow direction and vorticity computed from SLP on a 16-point grid,
yielding 26 types (8 directional, 16 hybrid, 2 rotational) collapsed to 10 by
splitting each hybrid half-and-half between its parents. Its own subject is
winter *precipitation*, so its fire relevance is entirely indirect. Two things
carry over.

**The summer baseline.** Summer is dominated by the **NE and N** types — an
extended Azores high producing persistent northerlies along the coast — while
**S and SE are effectively zero from May to August**. Any claim that a
southerly-flow pattern drives Portuguese fire is therefore a claim about a rare
situation, not a typical one.

**A direct criticism of this repo's method.** From the introduction:

> PCA results may vary significantly both with dataset length and with size of
> the window used, whereas k-means cluster analysis requires a predefinition of
> the number of clusters to be retained, giving a wide range of results for the
> same area of study.

It cites Portuguese studies landing anywhere from 4 to 13 circulation types.
That is exactly the instability behind our own finding that no *k* from 2 to 10
clears a red-noise null in either domain — we already report the partition as a
useful discretisation rather than a discovered set of states, and this is the
citation for why that caveat is mandatory rather than modest.

---

## Carmo et al. (2022) — the paper this project is closest to

IPMA's climatology of extreme wildfires, 1980–2018, applying the Trigo CWT
scheme to ERA5 SLP.

**Extreme Wildfire Periods (EWP).** Daily burned area is segmented by
changepoint detection (PELT), and periods averaging ≥3 000 ha·day⁻¹ are
retained. Result: **53 periods, 392 days, 52% of all burned area**, mean
duration **7.4 days**, 75% of them in July–August. Note the unit is a
multi-day *period*, not an event and not a day — the same choice we arrive at
in `--step lead`, for the same reason.

**Table 1 — observed over expected critical days, all 26 types.** The abstract
names only NE and E; the table shows a coherent *suppressive* family too, which
is the half that matters most to us.

| type | summer | EWP | obs/exp | p |
|---|---|---|---|---|
| **E** | 3.0% | 9.7% | **3.19** | <.001 |
| **S** | 0.3% | 1.5% | 4.6 (n=6) | <.001 |
| **NE** | 22.1% | 38.0% | **1.72** | <.001 |
| C | 5.4% | 8.2% | 1.51 | .016 |
| N | 17.8% | 13.0% | 0.73 | .014 |
| AN | 7.4% | 4.1% | 0.55 | .011 |
| NW | 6.5% | 3.3% | 0.51 | .011 |
| **A** | 13.8% | 6.4% | **0.46** | <.001 |
| ANW | 3.8% | 1.5% | 0.40 | .017 |
| W | 3.7% | 0.8% | 0.21 | .002 |
| AW | 1.6% | 0.0% | 0.00 | .011 |

**Table 2 — the result the abstract does not state.** Z500 over Portugal, per
type, summer mean against EWP days:

| | NE | E | N | A | C |
|---|---|---|---|---|---|
| summer anomaly | +23.8 | +60.5 | −19.6 | +32.6 | −28.6 |
| **EWP anomaly** | **+64.1** | **+81.8** | **+33.6** | **+78.6** | **+31.5** |

On critical days **every** type shows positive Z500 over Portugal — including
the cyclonic type, which is −28.6 in the summer mean and flips to +31.5. The
type alone does not identify a critical day; the type *plus the height over
Portugal* does. Also: of 145 easterly summer days, 26% fall in critical periods
and **54% had daily burned area above 500 ha**.

Other findings: antecedent drought is necessary but **not sufficient**, with
short-range water stress mattering more than seasonal; **no trend** in summer
synoptic pattern frequency 1980–2018.

**The second pathway.** Rare in frequency, decisive in impact — cyclogenesis
west and northwest of Iberia drawing air from the southern quadrant:

> Cyclonic large-scale circulation presents a low-pressure centre over the
> Portuguese west coast. In the mid-troposphere, there is a well-defined inverse
> omega shape that leads to a closed cyclonic circulation in the southern part.

This is the mode behind August 2003, June 2017 and October 2017. They close by
flagging **upper-tropospheric lows cut off from the western stream** as the
open research question.

---

## What this means for this repository

### It independently confirms `--step maps`, including the falsification

**Trigo Figure A1 is the direct evidence.** It draws the SLP anomaly field of
each type, and the NE and E types — Carmo's two fire drivers — are defined by a
positive anomaly centred near **55°N, 10–15°W**, reaching +16 hPa, with a
negative anomaly to the south over Iberia and NW Africa. Our regime 2 puts its
ridge at **55°N 14°W** with a low to the south. Same dipole, same place, from
SLP-vorticity typing rather than EOF/k-means.

Note what Carmo's Figure 5 can and cannot settle: its domain stops near **50°N
and 25°W**, so a ridge centred at 55°N falls outside it entirely. That figure
shows the southern half of the dipole and nothing else. The northern half rests
on Trigo Figure A1 and on Carmo's phrase *"between the Azores and the British
Isles"*.

Neither paper puts the fire-driving ridge over **NW Africa**. The premise at the
top of `CLAUDE.md` was wrong, and the literature already showed it was wrong.

The suppressive side lines up as a family, not a single type: W 0.21, ANW 0.40,
A 0.46, NW 0.51, AN 0.55 — the westerly/anticyclonic maritime group, suppressed
by factors of two to five. Our regime 0 is a trough northwest of Iberia driving
exactly that flow, and sits at 0.47 over a five-day window. Same direction and
same order of magnitude; different quantities, so do not report them as equal.

### Table 2 reproduces here, and it quantifies the regime-2 problem

Carmo's finding that every type turns positive over Portugal on critical days
has a direct analogue in our data. Severe-day rate within each regime, split by
terciles of Z500 over a Portugal box:

| regime | low third | high third | ratio |
|---|---|---|---|
| 0 | 0.010 | 0.057 | 6.0 |
| 1 | 0.099 | 0.103 | 1.0 |
| **2** | **0.055** | **0.279** | **5.1** |
| 3 | 0.097 | 0.179 | 1.9 |

Regime 2's own mean is not what makes it dangerous: its top height-tercile runs
at 0.279 against a 0.110 national baseline, its bottom third at 0.055 — *below*
baseline. The label is carrying two populations, which is exactly the blending
this file already suspected, now measured.

Two caveats that must travel with those numbers. Our anomalies have the daily
domain mean removed and a per-gridpoint trend taken out (invariant 1), so "Z500
over Portugal" here means *height relative to the domain that day*, a position
measure — it is **not** comparable in level to Carmo's absolute +64/+82/+32, and
the sign of our regime means should not be read against theirs. And high heights
over Portugal imply hot, dry, subsident conditions, so this may be FWI in
another coordinate system: invariant 10 applies, and the effect is not additional
to fire danger until `lead_ratio_by_stratum` says so.

### It exposes a real problem with regime 2

Carmo treats the **anticyclonic NE/E pathway** (common, moderate, 48% of
critical days) and the **cyclogenesis pathway** (rare, catastrophic) as
*separate* synoptic situations. Our four-cluster partition is coarser than their
ten types, and our regime 2 composite shows **both features at once**.

That is either a genuine omega-block pattern — their language supports it — or a
compositing artefact averaging two distinct situations into one mean. A
composite cannot distinguish them. It matters operationally: one pathway is
frequent and moderate, the other rare and catastrophic, and a blended mean
understates both.

**Test — and the height terciles above are most of it already.** Regime 2 splits
5:1 on height over Portugal, so it is demonstrably not one population. What is
still open is whether the split is the *NE/E versus C* distinction Carmo draws,
or simply amplitude within one pattern. Sub-cluster the regime-2 days and check
whether the two sub-composites reproduce their NE/E and C panels; if they do,
four regimes is too few for the elevating side however well it scores.

### The benchmark tests the domain but not the method

An earlier version of this file called `canonical_summer` "a partition designed
for the Euro-Atlantic winter" and therefore a straw man. **That was wrong.**
`config.py` follows Cassou et al. (2005), which is specifically the *summer*
Euro-Atlantic partition, and it is a legitimate seasonal baseline.

The real limitation is narrower and still worth fixing: `canonical_summer` and
`pt_tuned` are the same method — EOF truncation plus k-means — differing only in
domain. Comparing them tests the domain choice and nothing else. A different
*method* family is needed to test whether the machinery earns its complexity,
and the obvious candidate is the Jenkinson-Collison scheme these papers use,
which is cheap, validated for this region, and what IPMA would reach for.

That benchmark now exists as `--step cwt`, and **the k-means partition loses
it** — see the top of `CLAUDE.md`. The caveat there is load-bearing: it ran on
Z500 rather than MSLP.

### The precursor is two days, and it corroborates the *weak* half of `--step lead`

An earlier version of this file read the paper's text as onset predictability
"on the same 3–5 day scale we measure". **Figure 4 does not support that**, and
the figure is the authority. Reading the NE share day by day around onset:

| | day −4 | day −3 | day −2 | day −1 | day +1 | day +2 |
|---|---|---|---|---|---|---|
| NE share | ~17% | ~20% | ~46% | ~46% | ~55% | ~56% |

The summer mean for NE is 22.1%. So days −4 and −3 sit *at or below*
climatology — there is no precursor at all — and the whole signal appears at
**day −2**.

That is not a lead result for the hold signal. It is corroboration of something
else we found: the *elevating* side is short-lived. Our regime 2 is significant
at lead 0 and 1 and gone by lead 2, and its window ratio decays 1.62 → 1.42 →
not significant. Carmo's two-day NE build is the same statement from the other
direction. Both say the escalate signal lives on a ~2-day horizon, which is why
it did not survive weekly aggregation in `--step forecast`.

Our five-day hold signal has no counterpart in their analysis, because they
study the onset of extreme periods rather than the quiet intervals between
them. It is neither confirmed nor contradicted there.

### Smaller notes

- Their severity threshold (≥3 000 ha·day⁻¹ over a period) and ours (≥3 fires
  of ≥100 ha in a day) are different definitions of the same idea. Any
  comparison of effect sizes has to say so.
- "Drought necessary but not sufficient" is the same structure as our FWI
  stratification result, and supports keeping the two layers separate.
- No trend in pattern frequency since 1980 is a useful negative: it means our
  detrending invariant is about the thermal expansion of heights, not about
  drift in regime occupancy.

---

## The wider literature — leads, not established facts

The following came from a survey done outside this repository and **the
citations have not been checked against the papers themselves**. That caution is
not boilerplate: everything above the line was rewritten once already because it
had been written from text extractions rather than from the figures, and a claim
about the precursor lead turned out to be wrong. Treat this section as a
to-verify list.

**The argument has three legs and no bridge.** The literature supports
(1) regimes are a source of extended-range predictability, with an operational
limit around 14–20 days rather than truly seasonal; (2) circulation types
discriminate Iberian fire danger — Pereira et al. (2005), the Trigo scheme,
Carmo et al. (2022), plus fire-weather-type classifications in Ruffault et al.
(2020) and Rodrigues et al. (2020); and (3) regime-conditioned forecasting adds
value over forecasting the impact variable directly, and seasonal fire services
are already operating (Turco et al. with the Catalan SPIF). Nothing found closes
the loop: forecast regime frequencies at S2S lead, demonstrated against a named
Portuguese prevention decision. This project is that bridge, which is a reason
to build it and also a reason not to claim precedent for it.

**Tension worth resolving: Pereira et al. (2005).** They describe large
Portuguese summer fires under a ridge over Iberia *with* strongly meridional
flow advecting hot dry air from North Africa. Our regime 1 has the highest
heights over Portugal of the four (+40.4 gpm relative to the domain) and is
fire-**neutral** (OR 1.06) — and it is the only regime whose severe-day rate is
flat across height terciles (1.04, against 5.1 for regime 2). That is what a
conjunction looks like when only half of it is present: ridge without advection.

An attempt to test it here failed for a reason worth recording. Splitting each
regime by the sign of the southerly flow index gives nothing useful — regime 2's
southerly days run at 0.154 against 0.186 for its northerly days, the wrong
direction — because the indices are computed on Z500 and Pereira's claim is
about *surface* south-easterly advection. Upper-level flow direction is not a
proxy for it. The test needs MSLP.

**Two questions to check that ought to have answers already:** whether
Ruffault's fire-weather types have ever been mapped onto Euro-Atlantic regime
states, and whether Barbero or Moron have done regime-frequency to burned-area
work for France that could be mirrored here.

### What this changes about priorities

**MSLP is now the bottleneck for three separate things**, which is the strongest
argument yet for retrieving it: the CWT benchmark is currently run on the wrong
field, the Pereira conjunction cannot be tested without surface flow, and any
comparison against published CWT frequencies is meaningless until the
classification matches the published one.

**Cost–loss is the missing analysis, and it is cheap.** Every number in this
project is a rate or a ratio. An operational audience decides on expected
expense, and the extreme-precipitation work cited above makes its case on
cost–loss value at low ratios. The five-day hold signal has all the inputs
needed — hit rate, false-alarm rate, base rate — and converting it would say
whether a 0.47 ratio is worth acting on far better than the ratio itself does.
