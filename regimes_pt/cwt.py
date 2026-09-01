"""Jenkinson-Collison circulation weather types, in the Trigo & DaCâmara form.

The benchmark this project actually needs. `--step compare` scores the tuned
partition against `canonical_summer`, a k-means partition built for the
Euro-Atlantic winter, which is a straw man: it was never meant to resolve
Iberian summer circulation and nobody uses it operationally. The scheme here is
the one that *is* used - Trigo & DaCâmara (2000) adapted Jenkinson-Collison to
Portugal, and Carmo et al. (2022) used it to build the extreme-wildfire
climatology this project's fire layer is closest to. If an EOF/k-means
partition cannot beat it, the machinery is not earning its complexity.

Method: sixteen grid points around a central latitude and longitude give
finite-difference estimates of the geostrophic flow (westerly W, southerly S)
and of the shear vorticity (ZW + ZS). Flow direction plus the ratio of
vorticity to flow strength then sort each day into one of 26 types - eight pure
directional, two pure rotational, and sixteen hybrids.

Every rule compares |Z| against F and 2F, and W, S, ZW and ZS are all linear in
the input field, so the classification is invariant to the field's units and to
any constant scaling. That is what makes it possible to run the scheme on
geopotential height rather than on the mean sea-level pressure the original uses
- but note that this changes the *physics*, not just the units: 500 hPa flow is
not surface flow, and a type labelled "NE" here is an upper-level northeasterly,
not the surface northeasterly Carmo reports. Use MSLP for a like-for-like
comparison with the published frequencies.
"""

from __future__ import annotations

import numpy as np

# Trigo & DaCâmara place the grid on a 10 deg longitude by 5 deg latitude
# spacing; Carmo et al. centre it to span 25W-5E and 30-50N, which puts the
# central point over Portugal.
DLON, DLAT = 10.0, 5.0
CENTRE_LON, CENTRE_LAT = -10.0, 40.0

DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def grid_points(centre_lon: float = CENTRE_LON, centre_lat: float = CENTRE_LAT,
                dlon: float = DLON, dlat: float = DLAT):
    """The sixteen (lon, lat) pairs, in the canonical Jenkinson order.

        p1  p2                 rows at  +2*dlat
    p3  p4  p5  p6                      +1*dlat
    p7  p8  p9  p10                      0
   p11 p12 p13 p14                      -1*dlat
       p15 p16                          -2*dlat

    The order is not cosmetic - every coefficient below indexes into it.
    """
    lo = [centre_lon + k * dlon for k in (-1.5, -0.5, 0.5, 1.5)]
    la = [centre_lat + k * dlat for k in (2, 1, 0, -1, -2)]
    pts = [(lo[1], la[0]), (lo[2], la[0])]
    for j in (1, 2, 3):
        pts += [(lo[i], la[j]) for i in range(4)]
    pts += [(lo[1], la[4]), (lo[2], la[4])]
    return pts


def sample(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, **kw):
    """Pull the sixteen grid points out of a (time, nlat, nlon) field.

    Nearest-neighbour on purpose. The scheme is a finite-difference estimate on
    a coarse fixed stencil, so interpolating to exact degrees would imply a
    precision the method does not have; what matters is that the same points are
    used every day.
    """
    pts = grid_points(**kw)
    out = np.empty((field.shape[0], 16))
    for i, (plon, plat) in enumerate(pts):
        j = int(np.argmin(np.abs(lat - plat)))
        k = int(np.argmin(np.abs(lon - plon)))
        out[:, i] = field[:, j, k]
    return out


def indices(p: np.ndarray, centre_lat: float = CENTRE_LAT, dlat: float = DLAT):
    """Flow and vorticity indices from the sixteen sampled values.

    Returns `(W, S, Z, F)`: westerly flow, southerly flow, total shear
    vorticity, and resultant flow strength. Positive W is flow *from* the west,
    positive S is flow from the south, positive Z is cyclonic.
    """
    phi = np.deg2rad(centre_lat)
    sc = 1.0 / np.cos(phi)
    zwa = np.sin(phi) / np.sin(phi - np.deg2rad(dlat))
    zwb = np.sin(phi) / np.sin(phi + np.deg2rad(dlat))
    zsc = 1.0 / (2.0 * np.cos(phi) ** 2)

    def c(i):                      # 1-based, as in the published formulae
        return p[:, i - 1]

    w = 0.5 * (c(12) + c(13)) - 0.5 * (c(4) + c(5))
    s = sc * (0.25 * (c(5) + 2 * c(9) + c(13))
              - 0.25 * (c(4) + 2 * c(8) + c(12)))
    zw = (zwa * (0.5 * (c(15) + c(16)) - 0.5 * (c(8) + c(9)))
          - zwb * (0.5 * (c(8) + c(9)) - 0.5 * (c(1) + c(2))))
    zs = zsc * (0.25 * (c(6) + 2 * c(10) + c(14))
                - 0.25 * (c(5) + 2 * c(9) + c(13))
                - 0.25 * (c(4) + 2 * c(8) + c(12))
                + 0.25 * (c(3) + 2 * c(7) + c(11)))
    return w, s, zw + zs, np.sqrt(w ** 2 + s ** 2)


def direction(w: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Eight-point compass sector of the direction the flow comes *from*.

    Returned as an index into `DIRECTIONS`. Treating (W, S) as the wind vector's
    eastward and northward components, the meteorological "from" direction is
    270 deg minus the vector bearing; the sectors are 45 deg wide and centred on
    the compass points, so north wraps across 337.5/22.5.
    """
    deg = (270.0 - np.degrees(np.arctan2(s, w))) % 360.0
    return (np.floor((deg + 22.5) % 360.0 / 45.0).astype(int)) % 8


def classify(field: np.ndarray, lat: np.ndarray, lon: np.ndarray,
             centre_lon: float = CENTRE_LON, centre_lat: float = CENTRE_LAT,
             dlon: float = DLON, dlat: float = DLAT) -> np.ndarray:
    """Daily circulation weather type as a string label, one of 26.

    The rules are Trigo & DaCâmara's:
      |Z| < F        pure directional  (8)
      |Z| > 2F       pure rotational, C if Z > 0 else A  (2)
      F <= |Z| <= 2F hybrid, rotation prefixed to direction  (16)

    Unlike Jenkinson-Collison for the British Isles there is no unclassified
    class - the original paper disseminates the <2% of ambiguous cases among the
    retained types, and days sitting exactly on a boundary are assigned by the
    inequalities above rather than dropped.
    """
    p = sample(field, lat, lon, centre_lon=centre_lon, centre_lat=centre_lat,
               dlon=dlon, dlat=dlat)
    w, s, z, f = indices(p, centre_lat=centre_lat, dlat=dlat)
    d = direction(w, s)
    az = np.abs(z)

    labels = np.empty(len(w), dtype=object)
    pure_dir = az < f
    pure_rot = az > 2 * f
    hybrid = ~pure_dir & ~pure_rot
    rot = np.where(z > 0, "C", "A")

    for i in range(len(w)):
        if pure_dir[i]:
            labels[i] = DIRECTIONS[d[i]]
        elif pure_rot[i]:
            labels[i] = rot[i]
        else:
            labels[i] = rot[i] + DIRECTIONS[d[i]]
    return labels


def to_codes(labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Map string labels to integer codes plus the ordered vocabulary.

    `fire_link` works in integer regime codes throughout, so anything that wants
    to reuse the odds-ratio or skill machinery needs this.
    """
    vocab = sorted(set(labels.tolist()))
    index = {name: i for i, name in enumerate(vocab)}
    return np.array([index[x] for x in labels], dtype=int), vocab
