"""Configuration: domains, seasons, preprocessing defaults.

Two competing domain definitions are provided so the canonical baseline and the
Portugal-tuned alternative can be run through an identical pipeline and compared
on the same skill metric.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class Domain:
    """A lat/lon box for regime definition.

    Attributes
    ----------
    name : short identifier used in filenames
    lon : (west, east) in degrees east, negative for west
    lat : (south, north) in degrees north
    note : provenance / rationale
    """
    name: str
    lon: Tuple[float, float]
    lat: Tuple[float, float]
    note: str = ""

    @property
    def cds_area(self):
        """CDS expects [North, West, South, East]."""
        return [self.lat[1], self.lon[0], self.lat[0], self.lon[1]]


# ---------------------------------------------------------------------------
# Domains
# ---------------------------------------------------------------------------

CANONICAL = Domain(
    name="canonical",
    lon=(-80.0, 40.0),
    lat=(20.0, 80.0),
    note=(
        "Euro-Atlantic domain used by Michelangeli et al. (1995), Cassou (2008) "
        "and the ECMWF operational weather-regime products. Baseline: keeps "
        "direct comparability with published S2S regime probability forecasts."
    ),
)

# Cassou et al. (2005) used a slightly narrower box for summer regimes.
CANONICAL_SUMMER = Domain(
    name="canonical_summer",
    lon=(-90.0, 30.0),
    lat=(20.0, 80.0),
    note="Summer Euro-Atlantic domain following Cassou et al. (2005).",
)

# The hypothesis under test: a tighter, Iberia-centred domain should let the
# ridge-over-NW-Africa + cut-off-low-west-of-Portugal configuration emerge as a
# cluster in its own right instead of being absorbed into 'Blocking'.
PT_TUNED = Domain(
    name="pt_tuned",
    lon=(-40.0, 10.0),
    lat=(25.0, 55.0),
    note=(
        "Portugal-tuned domain. Narrower and shifted south to resolve the "
        "sub-synoptic Iberian extreme-fire pattern. Loses direct compatibility "
        "with off-the-shelf S2S regime products - forecast fields must be "
        "projected onto these centroids locally."
    ),
)

DOMAINS = {d.name: d for d in (CANONICAL, CANONICAL_SUMMER, PT_TUNED)}


# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------

SEASONS = {
    "JJAS": (6, 7, 8, 9),      # Portuguese fire season, primary target
    "JJA": (6, 7, 8),
    "MJJASO": (5, 6, 7, 8, 9, 10),  # extended, captures Oct events (Ophelia 2017)
    "DJFM": (12, 1, 2, 3),     # winter, for the antecedent-drought layer
}


# ---------------------------------------------------------------------------
# Preprocessing / clustering defaults
# ---------------------------------------------------------------------------

@dataclass
class PreprocConfig:
    n_harmonics: int = 3          # harmonics retained in the daily climatology
    detrend: str = "both"         # 'none' | 'linear' | 'domain_mean' | 'both'
    lat_weight: bool = True       # sqrt(cos(lat)) area weighting
    var_frac: float = 0.90        # EOF truncation: retain this much variance
    max_eofs: int = 30            # hard cap on retained EOFs
    lowpass_days: int = 0         # 0 = none; >0 applies centred running mean


@dataclass
class ClusterConfig:
    k_range: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10)
    n_init: int = 100             # k-means restarts per fit
    n_partitions: int = 50        # partitions used for the classifiability index
    n_surrogates: int = 100       # red-noise surrogates for the significance test
    random_state: int = 20260817
    assign_corr_threshold: float = 0.40   # below this a day is 'unclassified'


@dataclass
class RunConfig:
    domain: str = "pt_tuned"
    season: str = "JJAS"
    year_start: int = 1980
    year_end: int = 2025
    level: int = 500              # hPa
    variable: str = "geopotential"
    preproc: PreprocConfig = field(default_factory=PreprocConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    data_dir: str = "./data"
    out_dir: str = "./out"
