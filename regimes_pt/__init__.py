"""Portugal-tuned weather-regime classification for Iberian fire circulation.

Deliberately empty of re-exports. Callers import submodules explicitly
(``from regimes_pt import cluster, preprocess``) so importing the package
costs nothing and pulls in no I/O dependency. ``download`` defers its cdsapi
and xarray imports to call time for the same reason: the test suites must run
with no network, no credentials and no I/O stack installed.
"""
