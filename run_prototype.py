#!/usr/bin/env python3
"""Run the canonical-vs-Portugal-tuned regime comparison on ERA5 + CEMS FWI.

    python run_prototype.py --step download
    python run_prototype.py --step regimes  --domain canonical_summer --k 4
    python run_prototype.py --step regimes  --domain pt_tuned --select-k
    python run_prototype.py --step compare

Requires a CDS account and ~/.cdsapirc. Expect the download of 46 JJAS seasons
of 1-degree Z500 to take a few hours in the CDS queue; it is done once.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

from regimes_pt import cluster, download, fire_link, preprocess
from regimes_pt.config import DOMAINS, RunConfig


def _regime_state_path(cfg, domain, k):
    return os.path.join(cfg.out_dir, f"regimes_{domain}_{cfg.season}_k{k}.pkl")


def step_download(cfg: RunConfig, args):
    for name in ("canonical_summer", "pt_tuned"):
        print(f"[{name}] Z500 ...")
        download.download_z500(cfg, DOMAINS[name])
    print("[FWI] CEMS reanalysis over mainland Portugal ...")
    download.download_fwi(cfg)
    print("done")


def _load_prepared(cfg: RunConfig, domain: str, target_grid: float = 1.0):
    import glob
    paths = sorted(glob.glob(os.path.join(
        cfg.data_dir, f"z{cfg.level}_{domain}_{cfg.season}_*.nc")))
    if not paths:
        raise FileNotFoundError(f"no Z500 files for domain '{domain}' - run --step download")
    da = download.open_z500(paths, level=cfg.level, target_grid=target_grid)
    p = cfg.preproc
    return preprocess.prepare_from_dataarray(
        da, n_harmonics=p.n_harmonics, detrend=p.detrend,
        lat_weight=p.lat_weight, var_frac=p.var_frac,
        max_eofs=p.max_eofs, lowpass_days=p.lowpass_days,
    )


def step_regimes(cfg: RunConfig, args):
    os.makedirs(cfg.out_dir, exist_ok=True)
    prep = _load_prepared(cfg, args.domain, args.grid)
    print(f"{args.domain}: {prep.anom.shape[0]} days, "
          f"{prep.eof.n_eof} EOFs ({100*prep.eof.explained.sum():.1f}% variance)")

    k = args.k
    if args.select_k:
        c = cfg.cluster
        print("\nClassifiability index vs red-noise null:")
        sel = cluster.select_k(
            prep.eof.pcs, k_range=c.k_range, n_partitions=c.n_partitions,
            n_surrogates=c.n_surrogates, n_init=10,
            random_state=c.random_state,
        )
        # The margin over the null is a more stable selection rule than bare
        # significance: on real data several k are usually significant at once.
        best = max(sel, key=lambda s: s.ci - s.ci_null_p95)
        print(f"\nlargest margin over the null at k={best.k}")
        with open(os.path.join(cfg.out_dir, f"ksel_{args.domain}.json"), "w") as f:
            json.dump([s.__dict__ for s in sel], f, indent=2)
        k = k or best.k

    labels_pc, cent_pc = cluster.fit_kmeans(
        prep.eof.pcs, k, n_init=cfg.cluster.n_init,
        random_state=cfg.cluster.random_state)
    patterns_w = cluster.centroids_to_patterns(cent_pc, prep.eof.eofs)
    assign = cluster.assign_days(prep.anom_w, patterns_w,
                                 corr_threshold=cfg.cluster.assign_corr_threshold)

    print(f"\nk={k}")
    print(f"  unclassified: {assign.unclassified_frac:.1%}")
    print(f"  frequency:    {np.round(cluster.regime_frequency(assign.labels, k), 3)}")
    print(f"  persistence:  {np.round(cluster.mean_persistence(assign.labels, k), 1)} days")

    state = dict(domain=args.domain, k=k, patterns_w=patterns_w,
                 labels=assign.labels, max_corr=assign.max_corr,
                 times=prep.times, anom_w=prep.anom_w, area_w=prep.area_w,
                 lat=prep.lat, lon=prep.lon, shape=prep.shape)
    with open(_regime_state_path(cfg, args.domain, k), "wb") as f:
        pickle.dump(state, f)
    print(f"  saved -> {_regime_state_path(cfg, args.domain, k)}")


def _load_fwi(cfg: RunConfig):
    import glob

    paths = sorted(glob.glob(os.path.join(cfg.data_dir, f"fwi_pt_{cfg.season}_*.nc")))
    if not paths:
        raise FileNotFoundError("no FWI files - run --step download")
    return download.open_fwi(paths)


def step_compare(cfg: RunConfig, args):
    fwi, ftimes = _load_fwi(cfg)
    region_labels = fire_link.regionalize(fwi, n_regions=args.n_regions)
    reg = fire_link.region_series(fwi, region_labels)
    print(f"FWI: {fwi.shape[1]} land points -> {args.n_regions} fire-climate zones")

    import glob
    results, per_region = [], {}
    for path in sorted(glob.glob(os.path.join(cfg.out_dir, f"regimes_*_{cfg.season}_k*.pkl"))):
        with open(path, "rb") as f:
            st = pickle.load(f)
        name = f"{st['domain']}_k{st['k']}"

        # Align the Z500 and FWI calendars before anything else.
        import pandas as pd
        zt = pd.to_datetime(st["times"])
        common = zt.intersection(ftimes)
        zi = pd.Index(zt).get_indexer(common)
        fi = pd.Index(ftimes).get_indexer(common)
        labels = st["labels"][zi]
        anom_w = st["anom_w"][zi]
        doy = common.dayofyear.values
        years = common.year.values

        print(f"\n=== {name} ({len(common)} aligned days) ===")
        rows = []
        for r in range(args.n_regions):
            exc = fire_link.exceedance(reg[fi, r], args.percentile, doy=doy)
            comp = fire_link.fire_day_composite(anom_w, exc)
            ms = fire_link.match_scores(comp, st["patterns_w"])
            odds = fire_link.odds_by_regime(labels, exc, st["k"],
                                            block_len=7, n_boot=2000)
            sk = fire_link.cv_brier_skill(labels, exc, years, st["k"])
            top = int(np.argmax([o.odds_ratio for o in odds]))
            print(f"  zone {r}: composite match r={ms.max():.3f} "
                  f"(centroid {int(np.argmax(ms))}) | "
                  f"top OR = {odds[top].odds_ratio:.2f} "
                  f"[{odds[top].ci_low:.2f},{odds[top].ci_high:.2f}] regime {top} | "
                  f"BSS {sk.bss:+.4f}")
            rows.append(dict(zone=r, match=float(ms.max()), bss=sk.bss,
                             auc=sk.auc, top_regime=top,
                             odds=odds[top].odds_ratio))

        per_region[name] = rows
        national = fire_link.exceedance(reg.mean(axis=1)[fi], args.percentile, doy=doy)
        results.append((name, fire_link.cv_brier_skill(labels, national, years, st["k"])))

    print("\n" + fire_link.compare_configurations(results))
    with open(os.path.join(cfg.out_dir, "comparison.json"), "w") as f:
        json.dump(per_region, f, indent=2, default=float)


def _load_fires(cfg: RunConfig):
    """ICNF occurrence record as (dates, area_ha, district), fires >= 100 ha.

    Not a download: the extract is a fixed input tracked in the repo, because
    every number in the fire layer depends on this exact file and it cannot be
    regenerated from any API.
    """
    import glob

    import pandas as pd

    paths = sorted(glob.glob(os.path.join(cfg.data_dir, "ocoPT_*.csv")))
    if not paths:
        raise FileNotFoundError(
            f"no ICNF occurrence CSV in {cfg.data_dir} (expected ocoPT_*.csv)")
    df = pd.read_csv(paths[0])
    df["date"] = pd.to_datetime(df["data_alerta"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def step_fires(cfg: RunConfig, args):
    """Regime-conditioned odds of an actual large-fire day.

    The layer CLAUDE.md deliberately left unbuilt, now that the ICNF record is
    available. Note what the target is: fires that reached 100 ha, not FWI
    exceedance. Regimes do not predict ignition - ignition here is
    overwhelmingly human - they predict whether an ignition escapes, and
    escape is what a circulation pattern can plausibly govern. Testing against
    an absolute FWI threshold instead finds nothing, which is how this layer
    was mistakenly written off before the fire record was in hand.
    """
    import glob
    import json

    import pandas as pd

    fires = _load_fires(cfg)
    fwi, ftimes = download.open_fwi(
        sorted(glob.glob(os.path.join(cfg.data_dir, f"fwi_pt_{cfg.season}_*.nc"))))
    national_fwi = pd.Series(fwi.mean(axis=1), index=ftimes)

    daily = fires.groupby("date").agg(n=("codigo", "size"),
                                      ha=("area_total_ha", "sum"))
    out = {"n_fires": int(len(fires)),
           "burned_ha": float(fires["area_total_ha"].sum()),
           "year_range": [int(fires.date.dt.year.min()),
                          int(fires.date.dt.year.max())],
           "partitions": {}}

    for path in sorted(glob.glob(os.path.join(
            cfg.out_dir, f"regimes_*_{cfg.season}_k*.pkl"))):
        with open(path, "rb") as fh:
            st = pickle.load(fh)
        name = f"{st['domain']}_k{st['k']}"
        k = st["k"]

        # Restrict the circulation record to the years the fire record covers.
        times = pd.to_datetime(st["times"])
        keep = ((times.year >= fires.date.dt.year.min())
                & (times.year <= fires.date.dt.year.max()))
        times, labels = times[keep], st["labels"][keep]

        d = pd.DataFrame(index=times).join(daily).fillna({"n": 0, "ha": 0.0})
        counts, burned = d.n.values, d.ha.values
        fwi_day = national_fwi.reindex(times).values

        outcomes = {
            "any_large_fire": counts > 0,
            "severe_day_3plus": counts >= 3,
            "extreme_day_1000ha": burned >= 1000.0,
        }

        entry = {"n_days": int(len(times)),
                 "unclassified_frac": float((labels < 0).mean()),
                 "outcomes": {}, "episodes": {}, "stand_down": {}}

        for oname, exc in outcomes.items():
            odds = fire_link.odds_by_regime(labels, exc, k, block_len=7,
                                            n_boot=args.n_boot)
            skill = fire_link.cv_brier_skill(labels, exc, times.year.values, k)
            entry["outcomes"][oname] = {
                "n_events": int(exc.sum()),
                "auc": skill.auc, "bss": skill.bss,
                "regimes": [{"regime": o.regime, "odds_ratio": o.odds_ratio,
                             "ci_low": o.ci_low, "ci_high": o.ci_high,
                             "n_days": o.n_days,
                             "excludes_one": bool(o.ci_low > 1 or o.ci_high < 1)}
                            for o in odds],
            }

        # Does the regime survive conditioning on the stronger predictor?
        finite = np.isfinite(fwi_day)
        quartile = np.full(len(fwi_day), -1)
        quartile[finite] = pd.qcut(fwi_day[finite], 4, labels=False)
        severe = outcomes["severe_day_3plus"]
        entry["fwi_quartile_rate"] = fire_link.stratified_rate(
            labels, quartile, severe, k, 4).tolist()

        for r in range(k):
            entry["episodes"][str(r)] = [int(x) for x in
                                         fire_link.episodes(labels, r)]
            sd = fire_link.stand_down(labels, severe, burned, r)
            entry["stand_down"][str(r)] = {
                "days": sd.days, "day_fraction": sd.day_fraction,
                "events_missed": sd.events_missed,
                "event_fraction": sd.event_fraction,
                "burned_fraction": sd.burned_fraction,
            }

        out["partitions"][name] = entry

        print(f"\n=== {name}  ({entry['n_days']} {cfg.season} days, "
              f"{100*entry['unclassified_frac']:.1f}% unclassified) ===")
        for oname, res in entry["outcomes"].items():
            print(f"  {oname:20s} n={res['n_events']:4d}  AUC {res['auc']:.3f}  "
                  f"BSS {res['bss']:+.4f}")
            for rr in res["regimes"]:
                mark = "*" if rr["excludes_one"] else " "
                print(f"      regime {rr['regime']}: OR {rr['odds_ratio']:5.2f} "
                      f"[{rr['ci_low']:5.2f},{rr['ci_high']:5.2f}]{mark}")
        print("  severe-day rate by FWI quartile x regime (rows Q1..Q4):")
        for qi, row in enumerate(entry["fwi_quartile_rate"]):
            cells = "  ".join("   -  " if x is None or np.isnan(x)
                              else f"{100*x:5.1f}%" for x in row)
            print(f"      Q{qi+1}  {cells}")

    path = os.path.join(cfg.out_dir, "fire_regime.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


def step_forecast(cfg: RunConfig, args):
    """How much forecast skill the fire signal requires to survive.

    Every odds ratio in the fire layer conditions on *observed* circulation. A
    planner acts on a forecast, so the operative question is how much of the
    signal survives an imperfect one. This needs no S2S archive: degrade the
    known labels, measure what is left, and compare the requirement against
    published sub-seasonal skill.

    Two products are tested separately because they are scored differently.
    A daily categorical label is what the three-state readiness ladder needs;
    a weekly regime *frequency* is what extended-range forecasts are actually
    verified on. Conflating them overstates the weaker one.
    """
    import glob
    import json

    import pandas as pd

    fires = _load_fires(cfg)
    daily = fires.groupby("date").agg(n=("codigo", "size"),
                                      ha=("area_total_ha", "sum"))
    rng = np.random.default_rng(cfg.cluster.random_state)
    path = _regime_state_path(cfg, args.domain, args.k or 4)
    with open(path, "rb") as fh:
        st = pickle.load(fh)
    k = st["k"]

    times = pd.to_datetime(st["times"])
    keep = ((times.year >= fires.date.dt.year.min())
            & (times.year <= fires.date.dt.year.max()))
    times, labels = times[keep], st["labels"][keep]
    d = pd.DataFrame(index=times).join(daily).fillna({"n": 0, "ha": 0.0})
    severe = (d.n.values >= 3)
    burned = d.ha.values
    season = times.year.values

    out = {"domain": st["domain"], "k": k, "n_days": int(len(times))}

    # Persistence: the floor any real forecast must beat.
    full_t = pd.to_datetime(st["times"]); full_lab = st["labels"]
    fy = full_t.year.values
    out["persistence"] = {}
    for lead in (1, 3, 5, 7, 10, 14, 21, 28, 35):
        a, b = full_lab[:-lead], full_lab[lead:]
        ok = (fy[:-lead] == fy[lead:]) & (a >= 0) & (b >= 0)
        out["persistence"][str(lead)] = float(np.mean(a[ok] == b[ok]))
    cats, counts = np.unique(labels, return_counts=True)
    out["climatological_hit_rate"] = float(((counts / counts.sum()) ** 2).sum())

    # Product A: daily categorical label.
    out["daily_categorical"] = []
    for alpha in (1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, .0):
        acc, ors = [], {r: [] for r in range(k)}
        for _ in range(args.n_draws):
            fc = fire_link.degrade_labels(labels, alpha, rng)
            acc.append(float(np.mean(fc == labels)))
            for r in range(k):
                ors[r].append(fire_link._odds_ratio(fc, severe, r)[0])
        out["daily_categorical"].append(
            {"alpha": alpha, "realized_accuracy": float(np.mean(acc)),
             "odds_ratio": {str(r): float(np.mean(v)) for r, v in ors.items()},
             "odds_ratio_sd": {str(r): float(np.std(v)) for r, v in ors.items()}})

    # Product B: weekly regime frequency, the quantity S2S is scored on.
    blocks = fire_link.block_aggregate(np.arange(len(times)), season, 7)
    freq0 = np.array([(labels[b] == 0).mean() for b in blocks])
    wk_severe = np.array([severe[b].sum() for b in blocks], dtype=float)
    wk_burn = np.array([burned[b].sum() for b in blocks])
    out["n_weeks"] = int(len(blocks))
    out["weekly_truth"] = {}
    order = np.argsort(np.argsort(freq0))
    ter = np.floor(3 * order / len(order)).astype(int)
    for g, nm in ((0, "low"), (1, "mid"), (2, "high")):
        m = ter == g
        out["weekly_truth"][nm] = {
            "severe_per_week": float(wk_severe[m].mean()),
            "burned_per_week": float(wk_burn[m].mean()),
            "weeks_with_severe": float((wk_severe[m] > 0).mean())}
    out["weekly_correlation"] = float(np.corrcoef(freq0, wk_severe)[0, 1])

    out["weekly_degraded"] = []
    for rho in (1.0, .8, .6, .5, .4, .3, .2):
        ratios = []
        for _ in range(args.n_draws * 5):
            fc = fire_link.degrade_series(freq0, rho, rng)
            o = np.floor(3 * np.argsort(np.argsort(fc)) / len(fc)).astype(int)
            hi, lo = wk_severe[o == 2].mean(), wk_severe[o == 0].mean()
            if lo > 0:
                ratios.append(hi / lo)
        out["weekly_degraded"].append(
            {"rho": rho, "high_over_low": float(np.mean(ratios)),
             "sd": float(np.std(ratios))})

    print(f"=== forecast penalty: {st['domain']} k={k} ===")
    print(f"climatological hit rate {100*out['climatological_hit_rate']:.1f}%; "
          f"persistence at 14d {100*out['persistence']['14']:.1f}%, "
          f"28d {100*out['persistence']['28']:.1f}%")
    print("\ndaily categorical label - realized accuracy -> odds ratio")
    for row in out["daily_categorical"]:
        cells = "  ".join(f"r{r}={row['odds_ratio'][str(r)]:.2f}" for r in range(k))
        print(f"  acc {100*row['realized_accuracy']:5.1f}%   {cells}")
    print(f"\nweekly regime-0 frequency ({out['n_weeks']} weeks, "
          f"r={out['weekly_correlation']:+.3f} with severe-day count)")
    for nm in ("low", "mid", "high"):
        w = out["weekly_truth"][nm]
        print(f"  {nm:5s} severe/wk {w['severe_per_week']:.2f}   "
              f"burned/wk {w['burned_per_week']/1000:6.1f}k ha")
    print("\nweekly forecast correlation -> high/low tercile severe-day ratio")
    for row in out["weekly_degraded"]:
        print(f"  rho {row['rho']:.1f}   ratio {row['high_over_low']:.2f} "
              f"+-{row['sd']:.2f}")

    dest = os.path.join(cfg.out_dir, "forecast_penalty.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {dest}")


def step_lead(cfg: RunConfig, args):
    """How far ahead the regime signal carries with no forecast at all.

    `step_forecast` asks how much skill the signal needs and finds the answer
    is more than sub-seasonal forecasting has. This asks the complementary
    question: today's regime is *analysed*, so how many days of usable lead
    does it buy for free? That reframes the product from a 2-6 week
    pre-positioning aid - the operational context CLAUDE.md assumes, which the
    forecast penalty rules out - to a short-range modifier, which needs no
    regime forecast skill whatsoever.

    Two results decide whether the layer is worth operating. The horizon scan
    gives the window over which the signal is still significant. The
    FWI-stratified table gives the only thing that matters after that: whether
    the regime adds anything to the fire-danger forecast ICNF already uses, or
    merely restates it.
    """
    import glob
    import json

    import pandas as pd

    fires = _load_fires(cfg)
    daily = fires.groupby("date").agg(n=("codigo", "size"),
                                      ha=("area_total_ha", "sum"))
    path = _regime_state_path(cfg, args.domain, args.k or 4)
    with open(path, "rb") as fh:
        st = pickle.load(fh)
    k = st["k"]
    times = pd.to_datetime(st["times"])
    labels = st["labels"]

    # FWI comes from its own archive on its own calendar; intersect, never
    # assume alignment (see CLAUDE.md - different products, different leap-day
    # and time-zone handling).
    fwi_paths = sorted(glob.glob(os.path.join(cfg.data_dir, "fwi_*.nc")))
    if not fwi_paths:
        raise FileNotFoundError(f"no CEMS FWI files in {cfg.data_dir}")
    values, fwi_times = download.open_fwi(fwi_paths)
    fwi = pd.Series(values.mean(axis=1), index=pd.to_datetime(fwi_times))

    keep = (times.isin(fwi.index)
            & (times.year >= fires.date.dt.year.min())
            & (times.year <= fires.date.dt.year.max()))
    times, labels = times[keep], labels[keep]
    fwi_daily = fwi.reindex(times).values
    d = pd.DataFrame(index=times).join(daily).fillna({"n": 0, "ha": 0.0})
    severe = (d.n.values >= 3)
    season = times.year.values

    regime = args.regime
    out = {"domain": st["domain"], "k": k, "regime": regime,
           "n_days": int(len(times)),
           "years": [int(times.year.min()), int(times.year.max())]}

    out["horizons"] = []
    for h in (1, 2, 3, 5, 7, 10, 14):
        r = fire_link.lead_ratio(labels, severe, season, regime, h,
                                 n_boot=args.n_boot)
        out["horizons"].append({
            "horizon": h, "rate_in": r.rate_in, "rate_out": r.rate_out,
            "ratio": r.ratio, "ci": [r.ci_low, r.ci_high], "n_in": r.n_in})

    strat = fire_link.lead_ratio_by_stratum(
        labels, severe, fwi_daily, season, regime, args.horizon,
        n_strata=args.n_strata, n_boot=args.n_boot)
    out["fwi_strata"] = [
        {"stratum": s.stratum, "rate_in": s.rate_in, "rate_out": s.rate_out,
         "ratio": s.ratio, "ci": [s.ci_low, s.ci_high], "n_in": s.n_in}
        for s in strat]

    print(f"=== free lead: {st['domain']} k={k} regime {regime} "
          f"({out['years'][0]}-{out['years'][1]}, {out['n_days']} days) ===")
    print("conditioning day is analysed, not forecast - this lead costs no skill\n")
    print("  window     rate|r    rate|other   ratio  [95% CI]")
    for row in out["horizons"]:
        lo, hi = row["ci"]
        mark = "*" if (lo > 1 or hi < 1) else " "
        print(f"  +1..{row['horizon']:2d}d   {row['rate_in']:6.3f}    "
              f"{row['rate_out']:8.3f}   {row['ratio']:5.2f} "
              f"[{lo:4.2f},{hi:4.2f}]{mark}")

    print(f"\nwithin quantile bins of {args.horizon}-day window-mean FWI "
          f"- does the regime add to fire danger, or restate it?")
    print("  stratum    n(r)   rate|r    rate|other   ratio  [95% CI]")
    for row in out["fwi_strata"]:
        lo, hi = row["ci"]
        mark = "*" if (lo > 1 or hi < 1) else " "
        name = f"Q{row['stratum'] + 1}"
        print(f"  {name:8s} {row['n_in']:5d}  {row['rate_in']:6.3f}    "
              f"{row['rate_out']:8.3f}   {row['ratio']:5.2f} "
              f"[{lo:4.2f},{hi:4.2f}]{mark}")
    print("\n  * = 95% block-bootstrap CI excludes 1")

    dest = os.path.join(cfg.out_dir, "free_lead.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {dest}")


def step_maps(cfg: RunConfig, args):
    """Draw the regimes. The one claim in this project nothing else can test.

    Every result so far is a number conditioned on a label. Whether those
    labels correspond to the circulation the tuned domain was drawn to resolve
    - an upper ridge over NW Africa with a cut-off low west of Portugal - is
    not something an odds ratio can answer, and it has never been looked at.

    Anomalies are recomputed rather than read from the state pickle so that
    the map is in gpm on the unweighted grid: the stored `anom_w` carries the
    sqrt(cos lat) factor, which belongs in the distance metric and nowhere
    near a plot.
    """
    import glob

    from regimes_pt import plots

    path = _regime_state_path(cfg, args.domain, args.k or 4)
    with open(path, "rb") as fh:
        st = pickle.load(fh)
    k = st["k"]
    prep = _load_prepared(cfg, args.domain, args.grid)
    if len(prep.times) != len(st["labels"]):
        raise ValueError(
            f"prepared field has {len(prep.times)} days but the fitted state has "
            f"{len(st['labels'])} - refit with --step regimes before mapping")

    # Seasonal-mean total height, so the contours show closed features rather
    # than an anomaly pattern floating free of the mean flow.
    paths = sorted(glob.glob(os.path.join(
        cfg.data_dir, f"z{cfg.level}_{args.domain}_{cfg.season}_*.nc")))
    da = download.open_z500(paths, level=cfg.level, target_grid=args.grid)
    total_mean = da.mean("time").values.ravel()

    # Carry the fire result onto the map, so the pattern and the risk it
    # implies are read together rather than in separate documents.
    notes = {}
    fire_json = os.path.join(cfg.out_dir, "fire_regime.json")
    if os.path.exists(fire_json):
        with open(fire_json) as fh:
            fr = json.load(fh)
        for row in fr.get("odds", []):
            notes[int(row["regime"])] = f"OR {row['odds_ratio']:.2f}"

    dest = os.path.join(cfg.out_dir, f"composites_{args.domain}_k{k}.png")
    plots.regime_panels(
        prep.anom, st["labels"], prep.lat, prep.lon, prep.shape, k, dest,
        total_mean=total_mean, annotations=notes,
        title=f"Z500 composites - {args.domain}, {cfg.season} 1980-2025, k={k}")
    print(f"wrote {dest}")

    for r in range(k):
        mean, t, n_eff = plots.composite(prep.anom, st["labels"], r)
        m = mean.reshape(prep.shape)
        i_hi = np.unravel_index(np.argmax(m), m.shape)
        i_lo = np.unravel_index(np.argmin(m), m.shape)
        print(f"  regime {r}: n_eff {n_eff:5.0f}   "
              f"max {m[i_hi]:+6.1f} gpm at {prep.lat[i_hi[0]]:.0f}N "
              f"{prep.lon[i_hi[1]]:.0f}E   "
              f"min {m[i_lo]:+6.1f} gpm at {prep.lat[i_lo[0]]:.0f}N "
              f"{prep.lon[i_lo[1]]:.0f}E   "
              f"|t|>2 on {100 * np.mean(np.abs(t) > 2):.0f}% of the domain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True,
                    choices=["download", "regimes", "compare", "fires",
                             "forecast", "lead", "maps"])
    ap.add_argument("--domain", default="pt_tuned", choices=list(DOMAINS))
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--select-k", action="store_true")
    ap.add_argument("--grid", type=float, default=1.0)
    ap.add_argument("--n-regions", type=int, default=6)
    ap.add_argument("--percentile", type=float, default=95.0)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", default="./out")
    # Narrowing the year range is the only cheap way to test a CDS request
    # before committing to the full 1980-2025 pull: 46 years x 2 datasets is
    # ~92 queued requests. Retrieval is idempotent - download_z500 and
    # download_fwi skip files that already exist - so a one-year smoke test
    # costs nothing and its output is reused by the full run.
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-draws", type=int, default=30)
    # --step lead. Regime 0 is the suppressive regime and the only one whose
    # signal survives aggregation; 5 days is the window it is still strong at
    # while remaining long enough to move crews on.
    ap.add_argument("--regime", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--n-strata", type=int, default=4)
    ap.add_argument("--year-start", type=int, default=None)
    ap.add_argument("--year-end", type=int, default=None)
    args = ap.parse_args()

    years = {
        name: value
        for name, value in (("year_start", args.year_start),
                            ("year_end", args.year_end))
        if value is not None
    }
    cfg = RunConfig(data_dir=args.data_dir, out_dir=args.out_dir, **years)
    os.makedirs(cfg.out_dir, exist_ok=True)
    {"download": step_download, "regimes": step_regimes,
     "compare": step_compare, "fires": step_fires,
     "forecast": step_forecast, "lead": step_lead,
     "maps": step_maps}[args.step](cfg, args)


if __name__ == "__main__":
    main()
