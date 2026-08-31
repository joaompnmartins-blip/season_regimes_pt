"""Composite maps of the fitted regimes.

Kept out of the numerical core deliberately: matplotlib and cartopy are heavy,
optional, and irrelevant to every invariant the pipeline is tested on. Nothing
here computes a statistic that another module does not already own - this is a
rendering layer over `preprocess` and `cluster` output, so that a broken plot
can never be mistaken for a broken result.

The maps answer a question no number in this project can. The Portugal-tuned
domain was drawn to resolve a specific configuration - an upper ridge over NW
Africa with a cut-off low west of Portugal - and until the composites are drawn
that claim rests on the domain's design intent rather than on evidence.
"""

from __future__ import annotations

import numpy as np

# Geopotential height contours. Anomalies are small next to the mean field, so
# the two need different intervals: 60 gpm resolves the ridge and trough axes
# of the total field without burying the map in ink.
TOTAL_INTERVAL = 60
ANOM_LEVELS = np.arange(-90, 91, 15)

# Portugal, for orientation. The mainland only - the archipelagos sit outside
# every domain used here.
PT_LON, PT_LAT = -8.2, 39.7


def effective_n(labels: np.ndarray, regime: int) -> float:
    """Independent-episode count for a regime, not its raw day count.

    A regime occupies runs of several days, so treating each day as an
    independent sample overstates the sample size by the mean run length and
    makes every composite look significant. This is the same correction the
    block bootstrap applies to the odds ratios, in the form a t-statistic
    needs.
    """
    from . import fire_link

    runs = fire_link.episodes(labels, regime)
    if runs.size == 0:
        return 0.0
    return float(runs.sum() / runs.mean())


def composite(anom: np.ndarray, labels: np.ndarray, regime: int):
    """Mean anomaly for a regime with its autocorrelation-aware t-statistic.

    Returns `(mean, t, n_eff)` in the physical units of `anom` - gpm, if the
    caller has already undone the area weighting. Do not pass `anom_w`: the
    sqrt(cos lat) factor is there to make Euclidean distance approximate an
    area-weighted metric, and it has no business in a map.
    """
    inr = labels == regime
    if not inr.any():
        raise ValueError(f"no days assigned to regime {regime}")
    x = anom[inr]
    mean = x.mean(axis=0)
    n_eff = effective_n(labels, regime)
    se = x.std(axis=0, ddof=1) / np.sqrt(max(n_eff, 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, mean / se, 0.0)
    return mean, t, n_eff


def regime_panels(anom: np.ndarray, labels: np.ndarray, lat: np.ndarray,
                  lon: np.ndarray, shape: tuple, k: int, dest: str,
                  total_mean: np.ndarray | None = None,
                  annotations: dict[int, str] | None = None,
                  title: str = "", t_crit: float = 2.0):
    """Draw one panel per regime: anomaly shaded, total height contoured.

    The total field is contoured as well as the anomaly because the physical
    claim is about *closed* features. A cut-off low is a closed contour in the
    height field; an anomaly map shows only a negative centre, which a plain
    trough would produce too. Plotting the anomaly alone would leave the
    central claim of the tuned domain unfalsifiable by eye.
    """
    import matplotlib
    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    proj = ccrs.PlateCarree()
    ncol = 2
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.1 * nrow),
                             subplot_kw={"projection": proj})
    axes = np.atleast_1d(axes).ravel()

    lon2, lat2 = np.meshgrid(lon, lat)
    cf = None
    for r in range(k):
        ax = axes[r]
        mean, t, n_eff = composite(anom, labels, r)
        m = mean.reshape(shape)

        cf = ax.contourf(lon2, lat2, m, levels=ANOM_LEVELS, cmap="RdBu_r",
                         extend="both", transform=proj)

        # Stipple where the composite is NOT resolved, rather than where it is.
        # Marking significance hides the weak areas under ink; marking the
        # unresolved areas leaves the eye drawn to what the data supports.
        ax.contourf(lon2, lat2, np.abs(t.reshape(shape)), levels=[0, t_crit],
                    colors="none", hatches=["xxx"], transform=proj)

        if total_mean is not None:
            tot = (total_mean + mean).reshape(shape)
            lo = TOTAL_INTERVAL * np.floor(tot.min() / TOTAL_INTERVAL)
            hi = TOTAL_INTERVAL * np.ceil(tot.max() / TOTAL_INTERVAL)
            cs = ax.contour(lon2, lat2, tot,
                            levels=np.arange(lo, hi + 1, TOTAL_INTERVAL),
                            colors="k", linewidths=.6, transform=proj)
            ax.clabel(cs, inline=True, fontsize=6, fmt="%d")

        ax.add_feature(cfeature.COASTLINE, linewidth=.5, edgecolor="#444")
        ax.add_feature(cfeature.BORDERS, linewidth=.3, edgecolor="#888")
        ax.plot(PT_LON, PT_LAT, marker="o", markersize=5, color="#111",
                markerfacecolor="none", markeredgewidth=1.2, transform=proj)
        gl = ax.gridlines(draw_labels=True, linewidth=.3, color="#bbb",
                          alpha=.6, linestyle=":")
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 7}

        note = (annotations or {}).get(r, "")
        ax.set_title(f"Regime {r}   n={int((labels == r).sum())} d "
                     f"(n_eff={n_eff:.0f}){'   ' + note if note else ''}",
                     fontsize=9, loc="left")

    for ax in axes[k:]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=11, y=.995)
    cb = fig.colorbar(cf, ax=axes[:k].tolist(), orientation="horizontal",
                      fraction=.045, pad=.06, shrink=.55)
    cb.set_label("Z500 anomaly (gpm).  Hatched: |t| < "
                 f"{t_crit:g} on independent episodes.  "
                 "Contours: total height, 60 gpm.", fontsize=8)
    fig.savefig(dest, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return dest
