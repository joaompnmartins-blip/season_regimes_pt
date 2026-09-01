# Related work

Notes on the two papers this prototype should be read against. Both are
Portuguese, both use the same circulation-typing tradition, and one of them
independently reaches the conclusion our composite maps reached in
`--step maps`.

The PDFs and their text extracts are **not tracked** (see `.gitignore`):
Trigo and DaCâmara (2000) is all-rights-reserved, and Carmo et al. (2022) is
CC BY-NC-**ND**, which a text extraction arguably violates. Fetch them from the
DOIs below.

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

**Findings.**

| | |
|---|---|
| NE + E types | 48% of critical days, at 2× and 3× their summer frequency (p < .001) |
| Driving feature | anticyclones over the eastern Atlantic *between the Azores and the British Isles* |
| A (anticyclonic) | the only significantly **suppressive** type — 6.4% of critical days against 13.8% of summer days |
| C (cyclonic) | bimodal; eases conditions late in a period, but also opens the most catastrophic ones |
| Z500 over Portugal | +40 gpm from summer mean to EWP |
| Antecedent drought | necessary but **not sufficient**; short-range water stress matters more than seasonal |
| Trend 1980–2018 | none detected in summer synoptic pattern frequency |

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

Our regime 2 composite puts the ridge at **55°N 14°W** and a closed low at
**42°N 14°W** directly beneath it. Carmo's phrase for the fire-driving
anticyclone — *"the eastern Atlantic between the Azores and the British Isles"*
— describes that ridge position almost exactly, and their "inverse omega with
closed cyclonic circulation in the southern part" is the same geometry, reached
by a different classification method on a different fire dataset.

Neither paper puts the fire-driving ridge over **NW Africa**. The premise at the
top of `CLAUDE.md` was wrong, and the literature agrees it was wrong.

The suppressive side also lines up in magnitude: their A type is
under-represented in critical days by roughly a factor of two; our regime 0 sits
at 0.47 over a five-day window. Different quantities, same order — do not report
these as equal.

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

**Test:** sub-cluster the regime-2 days, or condition on the sign of the
meridional flow over Portugal, and check whether the burned-area distribution
within regime 2 is bimodal. If it is, four regimes is too few for the elevating
side, however well it scores.

### The benchmark is wrong

We currently compare `pt_tuned` against `canonical_summer` — a partition
designed for the Euro-Atlantic winter. That is a weak control. The real question
is whether an EOF/k-means partition beats **NE + E CWT frequency**, which is
cheap to compute, already validated for this region, and what IPMA would
actually reach for. Until that comparison exists, "the tuned domain does better"
is a claim against a straw man.

### The precursor result is a direct analogue of `--step lead`

Their Figure 4: the NE share peaks at **55% on the first day** of an EWP against
a 22% summer average, and the E share keeps climbing through **day 3**. That is
onset predictability on the same 3–5 day scale we measure, arrived at
independently. It is corroboration, not a citation for our number.

### Smaller notes

- Their severity threshold (≥3 000 ha·day⁻¹ over a period) and ours (≥3 fires
  of ≥100 ha in a day) are different definitions of the same idea. Any
  comparison of effect sizes has to say so.
- "Drought necessary but not sufficient" is the same structure as our FWI
  stratification result, and supports keeping the two layers separate.
- No trend in pattern frequency since 1980 is a useful negative: it means our
  detrending invariant is about the thermal expansion of heights, not about
  drift in regime occupancy.
