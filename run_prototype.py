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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True,
                    choices=["download", "regimes", "compare", "fires"])
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
     "compare": step_compare, "fires": step_fires}[args.step](cfg, args)


if __name__ == "__main__":
    main()
