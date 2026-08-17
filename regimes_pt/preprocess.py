"""Anomaly construction and EOF truncation.

Everything here operates on plain numpy so it can be unit-tested without any
network or file access. The xarray adapter is `prepare_from_dataarray`.

Three choices in this module matter more than they look:

1. DETRENDING. Summer Z500 over the Euro-Atlantic sector has a large positive
   trend from thermal expansion of the warming troposphere. If it is left in,
   k-means will happily hand you a "cluster" that is really just "recent years",
   and regime frequencies will show spurious trends. The default removes both
   the daily area-weighted domain mean and a per-gridpoint linear trend.

2. AREA WEIGHTING. Anomalies are multiplied by sqrt(cos(lat)) before the EOF
   step so that Euclidean distance in the truncated space approximates an
   area-weighted L2 distance on the sphere. Without it, high-latitude grid
   points are massively over-counted.

3. PC SCALING. The retained PCs are NOT standardised to unit variance. Leading
   PCs carry more physical amplitude and should dominate the distance metric;
   whitening them would give a grid-scale noise EOF the same weight as the NAO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Climatology and anomalies
# ---------------------------------------------------------------------------

def harmonic_climatology(
    values: np.ndarray,
    doy: np.ndarray,
    n_harmonics: int = 3,
    period: float = 365.25,
) -> np.ndarray:
    """Smooth seasonal cycle, evaluated at each sample's day-of-year.

    Parameters
    ----------
    values : (time, space) anomaly-free field
    doy    : (time,) day of year, 1-366
    n_harmonics : harmonics of the annual cycle to retain

    Returns
    -------
    (time, space) climatology aligned with `values`.

    If the day-of-year coverage spans less than ~200 days - which is the case
    whenever only a single season was downloaded - annual harmonics are badly
    conditioned, and a quadratic in day-of-year is fitted instead. This is
    reported via a runtime warning rather than failing silently.
    """
    doy = np.asarray(doy, dtype=float)
    span = doy.max() - doy.min()

    if span < 200:
        import warnings
        warnings.warn(
            f"day-of-year span is {span:.0f} days; fitting a quadratic seasonal "
            "cycle instead of annual harmonics. This is expected for "
            "single-season archives.",
            RuntimeWarning,
        )
        x = (doy - doy.mean()) / max(span, 1.0)
        design = np.column_stack([np.ones_like(x), x, x ** 2])
    else:
        cols = [np.ones_like(doy)]
        for h in range(1, n_harmonics + 1):
            cols.append(np.cos(2 * np.pi * h * doy / period))
            cols.append(np.sin(2 * np.pi * h * doy / period))
        design = np.column_stack(cols)

    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    return design @ coef


def latitude_weights(lat: np.ndarray) -> np.ndarray:
    """sqrt(cos(lat)) weights, clipped at the poles."""
    w = np.sqrt(np.clip(np.cos(np.deg2rad(np.asarray(lat, dtype=float))), 0.0, None))
    return w


def remove_domain_mean(anom: np.ndarray, area_w: np.ndarray) -> np.ndarray:
    """Subtract the area-weighted spatial mean of each day's field.

    This is the standard way to strip the thermodynamic expansion signal without
    assuming the trend is linear in time.
    """
    w = area_w / area_w.sum()
    return anom - (anom @ w)[:, None]


def linear_detrend(anom: np.ndarray, t: Optional[np.ndarray] = None) -> np.ndarray:
    """Remove a least-squares linear trend from each column."""
    n = anom.shape[0]
    t = np.arange(n, dtype=float) if t is None else np.asarray(t, dtype=float)
    tc = (t - t.mean()) / (t.std() if t.std() > 0 else 1.0)
    design = np.column_stack([np.ones_like(tc), tc])
    coef, *_ = np.linalg.lstsq(design, anom, rcond=None)
    return anom - design @ coef


def running_mean(anom: np.ndarray, window: int) -> np.ndarray:
    """Centred running mean along time, edges handled by shrinking the window."""
    if window <= 1:
        return anom
    n = anom.shape[0]
    half = window // 2
    out = np.empty_like(anom)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = anom[lo:hi].mean(axis=0)
    return out


# ---------------------------------------------------------------------------
# EOF truncation
# ---------------------------------------------------------------------------

@dataclass
class EOFResult:
    pcs: np.ndarray            # (time, n_eof) - NOT standardised
    eofs: np.ndarray           # (n_eof, space) - weighted space
    explained: np.ndarray      # (n_eof,) fraction of variance
    total_variance: float
    n_eof: int


def eof_truncate(anom_w: np.ndarray, var_frac: float = 0.90,
                 max_eofs: int = 30) -> EOFResult:
    """Truncate the weighted anomaly field to its leading EOFs.

    Truncation is a denoising step: k-means on the raw grid is dominated by
    small-scale variance that has nothing to do with the large-scale flow, and
    the resulting partitions are far less reproducible.
    """
    X = np.asarray(anom_w, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    var = S ** 2
    total = var.sum()
    frac = var / total
    cum = np.cumsum(frac)

    n = int(np.searchsorted(cum, var_frac) + 1)
    n = max(2, min(n, max_eofs, len(S)))

    return EOFResult(
        pcs=U[:, :n] * S[:n],
        eofs=Vt[:n],
        explained=frac[:n],
        total_variance=float(total),
        n_eof=n,
    )


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

@dataclass
class PreparedField:
    anom: np.ndarray           # (time, space) unweighted anomalies, physical units
    anom_w: np.ndarray         # (time, space) area-weighted anomalies
    eof: EOFResult
    lat: np.ndarray
    lon: np.ndarray
    area_w: np.ndarray         # (space,) flattened sqrt(cos lat) weights
    times: np.ndarray
    shape: tuple               # (nlat, nlon) for reshaping back to maps

    def to_map(self, flat: np.ndarray) -> np.ndarray:
        """Reshape a flattened spatial vector back to (nlat, nlon)."""
        return np.asarray(flat).reshape(self.shape)

    def unweight(self, flat_w: np.ndarray) -> np.ndarray:
        """Undo the area weighting, e.g. to plot a centroid in gpm."""
        return np.asarray(flat_w) / self.area_w


def prepare(
    field: np.ndarray,          # (time, nlat, nlon)
    lat: np.ndarray,
    lon: np.ndarray,
    times: np.ndarray,
    doy: np.ndarray,
    n_harmonics: int = 3,
    detrend: str = "both",
    lat_weight: bool = True,
    var_frac: float = 0.90,
    max_eofs: int = 30,
    lowpass_days: int = 0,
) -> PreparedField:
    """Full anomaly + EOF pipeline. See module docstring for the rationale."""
    nt, nlat, nlon = field.shape
    X = field.reshape(nt, nlat * nlon).astype(float)

    clim = harmonic_climatology(X, doy, n_harmonics=n_harmonics)
    anom = X - clim

    area_w = (
        np.repeat(latitude_weights(lat), nlon)
        if lat_weight else np.ones(nlat * nlon)
    )

    if detrend in ("domain_mean", "both"):
        anom = remove_domain_mean(anom, area_w)
    if detrend in ("linear", "both"):
        anom = linear_detrend(anom)
    if lowpass_days > 1:
        anom = running_mean(anom, lowpass_days)

    anom_w = anom * area_w
    eof = eof_truncate(anom_w, var_frac=var_frac, max_eofs=max_eofs)

    return PreparedField(
        anom=anom, anom_w=anom_w, eof=eof,
        lat=np.asarray(lat), lon=np.asarray(lon),
        area_w=area_w, times=np.asarray(times), shape=(nlat, nlon),
    )


def prepare_from_dataarray(da, **kwargs) -> PreparedField:
    """Adapter for an xarray DataArray with dims (time, latitude, longitude)."""
    lat_name = "latitude" if "latitude" in da.dims else "lat"
    lon_name = "longitude" if "longitude" in da.dims else "lon"
    da = da.transpose("time", lat_name, lon_name)
    import pandas as pd
    t = pd.to_datetime(da["time"].values)
    return prepare(
        field=da.values,
        lat=da[lat_name].values,
        lon=da[lon_name].values,
        times=t.values,
        doy=t.dayofyear.values,
        **kwargs,
    )
