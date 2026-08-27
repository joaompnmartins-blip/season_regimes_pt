"""Retrieval of ERA5 Z500 and CEMS FWI reanalysis from the Copernicus CDS.

Requires a CDS account and ~/.cdsapirc. Neither dataset is available from a
sandboxed environment, so this module is written to be run on your own machine.

NOTE ON DATASET NAMES: the CDS catalogue is periodically restructured and
dataset identifiers / system versions drift. If a request 404s, check the
current identifier on the CDS web catalogue and update the constants below
rather than assuming the code is broken.
"""

from __future__ import annotations

import concurrent.futures as cf
import glob
import json
import os
import re
import urllib.request
from typing import Iterable, Optional, Sequence

from .config import Domain, RunConfig, SEASONS

# Two datastores, not one. ERA5 is on the CDS; the CEMS fire datasets were
# migrated to the Early Warning Data Store and are no longer served from the CDS
# at all - requesting cems-fire-historical-v1 there returns 404 "process not
# found", which reads like a drifted identifier but is a moved host.
#
# Verified 2026-08-17 against the live process endpoints of both stores.
CDS_URL = "https://cds.climate.copernicus.eu/api"
EWDS_URL = "https://ewds.climate.copernicus.eu/api"

ERA5_PL_DAILY = "derived-era5-pressure-levels-daily-statistics"
ERA5_PL_HOURLY = "reanalysis-era5-pressure-levels"
CEMS_FIRE = "cems-fire-historical-v1"


def _client(url: Optional[str] = None):
    """A cdsapi client, optionally pinned to a datastore other than the default.

    ~/.cdsapirc holds a single url/key pair, which covers the CDS only. EWDS is
    a separate store with its own licence acceptances; ECMWF single sign-on
    means the same token usually works for both, so the key from ~/.cdsapirc is
    reused unless EWDS_KEY overrides it.
    """
    try:
        import cdsapi
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "cdsapi not installed. `pip install cdsapi` and create ~/.cdsapirc"
        ) from exc

    if url is None:
        return cdsapi.Client()

    key = os.environ.get("EWDS_KEY") or cdsapi.Client().key
    return cdsapi.Client(url=url, key=key)


def _retrieve(client, dataset: str, request: dict, path: str) -> None:
    """Retrieve, translating a 404 into the datastore-split explanation.

    A bare 404 here is genuinely ambiguous - the identifier may have drifted, or
    the dataset may have moved store entirely - and the two have different fixes.
    """
    try:
        client.retrieve(dataset, request, path)
    except Exception as exc:
        text = str(exc)
        if "404" in text or "not found" in text.lower():
            raise RuntimeError(
                f"{dataset!r} not found on {getattr(client, 'url', 'the default store')}.\n"
                f"CEMS datasets live on the EWDS ({EWDS_URL}), ERA5 on the CDS\n"
                f"({CDS_URL}). If the host is right, the identifier has drifted:\n"
                "check the live catalogue and update the constants at the top of\n"
                "this module rather than assuming the code is broken."
            ) from exc
        raise


# One request per year is the slow way to do this. Queue wait dominates and is
# paid per request: the 2003 smoke test spent 43 of its 47 minutes queued and
# only 4 processing. Grouping years collapses that fixed cost, which is also
# ECMWF's own guidance - fewer, larger requests.
#
# 10 years x 4 months x 31 days x 1 level is ~1240 fields, comfortably inside
# the per-request limits. Raise it and you trade failure granularity for fewer
# queue waits: a rejected 46-year request loses everything, a rejected 10-year
# one loses a chunk.
# Fallbacks only - the real ceiling is read from the costing endpoint at run
# time by _years_per_request(). Measured 2026-08-26: cost is one unit per
# retrieved day, so a JJAS year costs 122 on both stores, but the ceiling is
# 400 on the CDS (3 JJAS years) and 3720 on the EWDS (30). A single constant
# would either overrun one store or starve the other by an order of magnitude.
YEARS_PER_REQUEST_CDS = 3
YEARS_PER_REQUEST_EWDS = 30

# Three concurrent workers produced a submit/reject loop against a queue that
# already held stuck jobs: CDS caps active requests per user, cdsapi retries a
# rejection by submitting a fresh request, and three workers doing that compound
# into a burst that keeps getting bounced. A serial run then completed cleanly.
#
# Two is the compromise - it halves the wall clock, which is dominated by queue
# wait (~55 min per ERA5 request against ~20 s on the EWDS), while staying well
# under whatever tripped the limiter. If rejections return, drop to 1: a
# rejected request never completes, so concurrency that provokes one is a net
# loss however fast it submits.
MAX_WORKERS = 2

_YEARS_IN_NAME = re.compile(r"_(\d{4})(?:-(\d{4}))?\.nc$")


def _years_on_disk(pattern: str) -> set:
    """Years already retrieved, parsed from filenames.

    Coverage is tracked by filename rather than by opening files, so resuming an
    interrupted run costs no I/O. Both `_2003.nc` and `_1980-1989.nc` count.
    """
    years = set()
    for path in glob.glob(pattern):
        m = _YEARS_IN_NAME.search(os.path.basename(path))
        if m:
            first = int(m.group(1))
            years.update(range(first, int(m.group(2) or first) + 1))
    return years


def _years_per_request(url: Optional[str], dataset: str,
                       one_year_request: dict, fallback: int) -> int:
    """Ask CADS how many years fit in one request, from a one-year probe.

    Request size is capped by a per-dataset "cost" limit, and exceeding it is a
    403 at submission - cheap to hit, but it fails the whole chunk. The limits
    are not published in the process schema and differ by store, so they are
    measured rather than assumed: cost scales linearly with the number of
    retrieved days, so limit // cost_of_one_year is the answer.

    Falls back to a measured constant if the endpoint is unreachable; being
    wrong here costs a retry, not correctness.
    """
    endpoint = (f"{url or CDS_URL}/retrieve/v1/processes/{dataset}"
                "/costing?request_origin=api")
    try:
        body = json.dumps({"inputs": one_year_request}).encode()
        req = urllib.request.Request(
            endpoint, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as fh:
            costing = json.load(fh)
        cost, limit = float(costing["cost"]), float(costing["limit"])
        if cost > 0:
            return max(1, int(limit // cost))
    except Exception:
        pass          # network, schema change, auth - all mean "use the default"
    return fallback


def _chunk(seq: Sequence[int], size: int):
    for i in range(0, len(seq), size):
        yield list(seq[i:i + size])


def _run_one(job) -> None:
    """Retrieve one request, writing through a .part file.

    An interrupted overnight run must not leave a truncated file behind: file
    existence is the resume mechanism, so a partial write would be silently
    treated as complete and poison the archive.
    """
    url, dataset, request, path = job
    tmp = path + ".part"
    _retrieve(_client(url), dataset, request, tmp)
    os.replace(tmp, path)


def _run_jobs(jobs: list, max_workers: int = MAX_WORKERS) -> None:
    """Submit retrievals concurrently, since queue wait overlaps.

    A client per job rather than one shared: cdsapi holds per-request state and
    is not documented as thread-safe.
    """
    if not jobs:
        return
    if len(jobs) == 1:
        _run_one(jobs[0])
        return

    with cf.ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as ex:
        futures = [ex.submit(_run_one, job) for job in jobs]
        for future in cf.as_completed(futures):
            future.result()   # re-raise the first failure, with its traceback


def _months(season: str) -> Sequence[str]:
    return [f"{m:02d}" for m in SEASONS[season]]


def _days() -> Sequence[str]:
    return [f"{d:02d}" for d in range(1, 32)]


def download_z500(
    cfg: RunConfig,
    domain: Domain,
    use_daily_product: bool = True,
    years_per_request: Optional[int] = None,
    max_workers: int = MAX_WORKERS,
) -> list[str]:
    """Download daily-mean geopotential at `cfg.level` hPa.

    Years are grouped as many to a request as the dataset's cost limit allows
    (queried at run time; pass `years_per_request` to override) and the
    resulting requests submitted concurrently, because CDS queue wait is per
    request and dominates the total. Years already covered by a file on disk
    are skipped, so an interrupted run resumes without refetching.

    No server-side regridding: CADS accepts no `grid` key for either ERA5
    pressure-level dataset (verified against both process schemas), so files
    arrive at native 0.25 deg. Coarsening to the 1 deg working grid happens at
    read time in `open_z500` - which matters, because 46 seasons of 0.25 deg
    Z500 over the canonical domain does not fit in RAM.

    Returns every path for this domain and season, old and new.
    """
    os.makedirs(cfg.data_dir, exist_ok=True)
    stem = f"z{cfg.level}_{domain.name}_{cfg.season}"
    pattern = os.path.join(cfg.data_dir, f"{stem}_*.nc")

    have = _years_on_disk(pattern)
    want = [y for y in range(cfg.year_start, cfg.year_end + 1) if y not in have]

    if not want:
        return sorted(glob.glob(pattern))

    def build(years):
        if use_daily_product:
            request = {
                "product_type": "reanalysis",
                "variable": [cfg.variable],
                "pressure_level": [str(cfg.level)],
                "year": [str(y) for y in years],
                "month": _months(cfg.season),
                "day": _days(),
                "daily_statistic": "daily_mean",
                "time_zone": "utc+00:00",
                "frequency": "1_hourly",
                "area": domain.cds_area,
            }
            dataset = ERA5_PL_DAILY
        else:
            # Fallback: 4x daily instantaneous, averaged downstream.
            request = {
                "product_type": "reanalysis",
                "variable": [cfg.variable],
                "pressure_level": [str(cfg.level)],
                "year": [str(y) for y in years],
                "month": _months(cfg.season),
                "day": _days(),
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "area": domain.cds_area,
                # CADS renamed the legacy `format` key; this dataset takes
                # `data_format` and rejects `format`.
                "data_format": "netcdf",
            }
            dataset = ERA5_PL_HOURLY
        return dataset, request

    dataset, _probe = build(want[:1])
    per = years_per_request or _years_per_request(
        None, dataset, _probe, YEARS_PER_REQUEST_CDS)

    jobs = []
    for years in _chunk(want, per):
        span = f"{years[0]}" if len(years) == 1 else f"{years[0]}-{years[-1]}"
        path = os.path.join(cfg.data_dir, f"{stem}_{span}.nc")
        jobs.append((None, dataset, build(years)[1], path))

    _run_jobs(jobs, max_workers)
    return sorted(glob.glob(pattern))


def download_fwi(
    cfg: RunConfig,
    area: Sequence[float] = (42.5, -10.0, 36.5, -6.0),  # N, W, S, E - mainland PT
    variables: Iterable[str] = ("fire_weather_index",),
    years_per_request: Optional[int] = None,
    max_workers: int = MAX_WORKERS,
) -> list[str]:
    """Download CEMS ERA5-based FWI reanalysis over mainland Portugal.

    Served from the EWDS, not the CDS. That store needs its own licence
    acceptance for this dataset - accepting the ERA5 licence on the CDS does
    not carry over, and the failure is a 403 at request time, not at login.

    Its cost ceiling is also far higher than the CDS one, so far more years fit
    in a single request; the limit is queried rather than assumed.

    The default box is deliberately a little larger than the coastline so that
    coastal grid points are not half-masked. Returns written paths.
    """
    os.makedirs(cfg.data_dir, exist_ok=True)
    stem = f"fwi_pt_{cfg.season}"
    pattern = os.path.join(cfg.data_dir, f"{stem}_*.nc")

    have = _years_on_disk(pattern)
    want = [y for y in range(cfg.year_start, cfg.year_end + 1) if y not in have]

    if not want:
        return sorted(glob.glob(pattern))

    def build(years):
        return {
            "product_type": "reanalysis",
            "variable": list(variables),
            "dataset_type": "consolidated_dataset",
            "system_version": "4_1",
            "year": [str(y) for y in years],
            "month": _months(cfg.season),
            "day": _days(),
            # Enum of strings on this dataset - '0.5/0.5' | '0.25/0.25' |
            # 'original_grid'. A ["0.25", "0.25"] pair is rejected.
            "grid": "0.25/0.25",
            "area": list(area),
            "data_format": "netcdf",
        }

    per = years_per_request or _years_per_request(
        EWDS_URL, CEMS_FIRE, build(want[:1]), YEARS_PER_REQUEST_EWDS)

    jobs = []
    for years in _chunk(want, per):
        span = f"{years[0]}" if len(years) == 1 else f"{years[0]}-{years[-1]}"
        path = os.path.join(cfg.data_dir, f"{stem}_{span}.nc")
        jobs.append((EWDS_URL, CEMS_FIRE, build(years), path))

    _run_jobs(jobs, max_workers)
    return sorted(glob.glob(pattern))


def _rename_time(da):
    """Normalise the time coordinate to `time`.

    Copernicus products disagree: some carry `valid_time`, some `time`,
    depending on dataset and vintage. Every reader must handle both - assuming
    one is how a reader works on ERA5 and then dies on CEMS.
    """
    if "valid_time" in da.dims or "valid_time" in da.coords:
        da = da.rename({"valid_time": "time"})
    return da


def _coarsen_to(da, target_grid: float):
    """Block-average from the native grid onto a `target_grid`-degree grid.

    Block mean rather than interpolation: regime patterns are planetary scale,
    so the area-mean of a block is the physically meaningful reduction, whereas
    subsampling would alias the sub-synoptic detail we are discarding on purpose.

    Applied lazily, before `.load()`, so the native-resolution array is never
    fully realised in memory - the point of the exercise.
    """
    lat = "latitude" if "latitude" in da.dims else "lat"
    lon = "longitude" if "longitude" in da.dims else "lon"
    if da[lat].size < 2:
        return da

    native = abs(float(da[lat][1] - da[lat][0]))
    factor = int(round(target_grid / native))
    if factor <= 1:
        return da
    return da.coarsen({lat: factor, lon: factor}, boundary="trim").mean()


def open_z500(paths: Sequence[str], level: int = 500,
              target_grid: Optional[float] = 1.0):
    """Open a multi-year Z500 archive as an (time, lat, lon) DataArray in gpm.

    ERA5 stores geopotential in m2 s-2; divided by g here so that composite maps
    are in geopotential metres, which is what everyone reads.

    `target_grid` defaults to 1 deg because CADS no longer regrids server-side
    and the files arrive at native 0.25 deg. Pass None to keep native
    resolution, but note that 46 seasons of 0.25 deg Z500 over the canonical
    domain will not fit in RAM - 1 deg is ample for planetary-scale regimes.
    """
    import xarray as xr

    ds = xr.open_mfdataset(paths, combine="by_coords")
    name = "z" if "z" in ds else list(ds.data_vars)[0]
    da = ds[name]

    if "pressure_level" in da.dims:
        da = da.sel(pressure_level=level, drop=True)
    elif "level" in da.dims:
        da = da.sel(level=level, drop=True)

    # Average sub-daily samples if the hourly fallback was used.
    da = _rename_time(da)
    if da.time.size and len(set(da.time.dt.floor("D").values)) < da.time.size:
        da = da.groupby("time.date").mean("time")
        da = da.rename({"date": "time"})
        da = da.assign_coords(time=[str(v) for v in da.time.values]).astype("float32")

    if target_grid is not None:
        da = _coarsen_to(da, target_grid)

    return (da / 9.80665).rename("z500_gpm").load()


def open_fwi(paths: Sequence[str]):
    """Open the CEMS FWI archive as (values[time, land_point], times).

    Returns numpy rather than a DataArray because everything downstream in
    fire_link is numpy, and the ocean mask has to be resolved here in any case:
    CEMS masks sea points as NaN, and k-means would otherwise cluster the mask
    rather than the fire climate.

    Note the variable is named `fwinx`, not `fire_weather_index` as requested -
    hence taking the sole data variable rather than looking one up by name.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    ds = xr.open_mfdataset(paths, combine="by_coords")
    da = _rename_time(ds[list(ds.data_vars)[0]]).load()

    lat = "latitude" if "latitude" in da.dims else "lat"
    lon = "longitude" if "longitude" in da.dims else "lon"
    da = da.transpose("time", lat, lon)

    values = da.values.reshape(da.shape[0], -1)
    land = ~np.isnan(values).any(axis=0)
    return values[:, land], pd.to_datetime(da["time"].values)
