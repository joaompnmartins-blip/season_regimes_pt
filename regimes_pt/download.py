"""Retrieval of ERA5 Z500 and CEMS FWI reanalysis from the Copernicus CDS.

Requires a CDS account and ~/.cdsapirc. Neither dataset is available from a
sandboxed environment, so this module is written to be run on your own machine.

NOTE ON DATASET NAMES: the CDS catalogue is periodically restructured and
dataset identifiers / system versions drift. If a request 404s, check the
current identifier on the CDS web catalogue and update the constants below
rather than assuming the code is broken.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

from .config import Domain, RunConfig, SEASONS

ERA5_PL_DAILY = "derived-era5-pressure-levels-daily-statistics"
ERA5_PL_HOURLY = "reanalysis-era5-pressure-levels"
CEMS_FIRE = "cems-fire-historical-v1"


def _client():
    try:
        import cdsapi
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "cdsapi not installed. `pip install cdsapi` and create ~/.cdsapirc"
        ) from exc
    return cdsapi.Client()


def _months(season: str) -> Sequence[str]:
    return [f"{m:02d}" for m in SEASONS[season]]


def _days() -> Sequence[str]:
    return [f"{d:02d}" for d in range(1, 32)]


def download_z500(
    cfg: RunConfig,
    domain: Domain,
    grid: float = 1.0,
    use_daily_product: bool = True,
) -> list[str]:
    """Download daily-mean geopotential at `cfg.level` hPa, one file per year.

    A 1.0 deg grid is ample for regime work: the patterns are planetary scale and
    a coarser grid substantially reduces both download volume and the dimension
    fed to the EOF step. Returns the list of written paths.
    """
    c = _client()
    os.makedirs(cfg.data_dir, exist_ok=True)
    paths = []

    for year in range(cfg.year_start, cfg.year_end + 1):
        path = os.path.join(
            cfg.data_dir, f"z{cfg.level}_{domain.name}_{cfg.season}_{year}.nc"
        )
        if os.path.exists(path):
            paths.append(path)
            continue

        if use_daily_product:
            request = {
                "product_type": "reanalysis",
                "variable": [cfg.variable],
                "pressure_level": [str(cfg.level)],
                "year": [str(year)],
                "month": _months(cfg.season),
                "day": _days(),
                "daily_statistic": "daily_mean",
                "time_zone": "utc+00:00",
                "frequency": "1_hourly",
                "area": domain.cds_area,
                "grid": [grid, grid],
            }
            dataset = ERA5_PL_DAILY
        else:
            # Fallback: 4x daily instantaneous, averaged downstream.
            request = {
                "product_type": "reanalysis",
                "variable": [cfg.variable],
                "pressure_level": [str(cfg.level)],
                "year": [str(year)],
                "month": _months(cfg.season),
                "day": _days(),
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "area": domain.cds_area,
                "grid": [grid, grid],
                "format": "netcdf",
            }
            dataset = ERA5_PL_HOURLY

        c.retrieve(dataset, request, path)
        paths.append(path)

    return paths


def download_fwi(
    cfg: RunConfig,
    area: Sequence[float] = (42.5, -10.0, 36.5, -6.0),  # N, W, S, E - mainland PT
    variables: Iterable[str] = ("fire_weather_index",),
) -> list[str]:
    """Download CEMS ERA5-based FWI reanalysis over mainland Portugal.

    The default box is deliberately a little larger than the coastline so that
    coastal grid points are not half-masked. Returns written paths.
    """
    c = _client()
    os.makedirs(cfg.data_dir, exist_ok=True)
    paths = []

    for year in range(cfg.year_start, cfg.year_end + 1):
        path = os.path.join(cfg.data_dir, f"fwi_pt_{cfg.season}_{year}.nc")
        if os.path.exists(path):
            paths.append(path)
            continue

        request = {
            "product_type": "reanalysis",
            "variable": list(variables),
            "dataset_type": "consolidated_dataset",
            "system_version": "4_1",
            "year": [str(year)],
            "month": _months(cfg.season),
            "day": _days(),
            "grid": ["0.25", "0.25"],
            "area": list(area),
            "data_format": "netcdf",
        }
        c.retrieve(CEMS_FIRE, request, path)
        paths.append(path)

    return paths


def open_z500(paths: Sequence[str], level: int = 500):
    """Open a multi-year Z500 archive as an (time, lat, lon) DataArray in gpm.

    ERA5 stores geopotential in m2 s-2; divided by g here so that composite maps
    are in geopotential metres, which is what everyone reads.
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
    if "valid_time" in da.dims:
        da = da.rename({"valid_time": "time"})
    if da.time.size and len(set(da.time.dt.floor("D").values)) < da.time.size:
        da = da.groupby("time.date").mean("time")
        da = da.rename({"date": "time"})
        da = da.assign_coords(time=[str(v) for v in da.time.values]).astype("float32")

    return (da / 9.80665).rename("z500_gpm").load()
