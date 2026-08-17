# Reproducible run environment for the SeasonRegimes_PT prototype.
#
# The image carries only code + dependencies. ERA5/CEMS netCDFs and fitted
# regime state live in bind-mounted volumes (see docker-compose.yml) because
# they are large and are not build inputs.

FROM python:3.12-slim

# numpy, scipy, scikit-learn, pandas, netCDF4 and matplotlib all publish
# manylinux wheels, so no HDF5/BLAS system packages are needed at build time.
# If you ever pin a dependency with no wheel for this platform, add
# build-essential here rather than switching off the slim base.

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Match the host UID/GID so files the container writes into the mounted data/
# and out/ directories are owned by you rather than by root.
ARG UID=1000
ARG GID=1000
RUN groupadd -g "${GID}" regimes \
 && useradd -u "${UID}" -g "${GID}" -m regimes

WORKDIR /app

# Dependencies first: this layer caches across code edits, and the scientific
# stack is by far the slowest part of the build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data /app/out && chown -R regimes:regimes /app

USER regimes

# No ENTRYPOINT on purpose: the image is a run environment, not a single
# command. The default shows the CLI; `docker compose run` overrides it to run
# the test suites or individual pipeline steps.
CMD ["python", "run_prototype.py", "--help"]
