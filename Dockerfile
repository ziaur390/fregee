# Multi-stage build for the mlserve-ops API.
#
# Three decisions worth stating:
#
# 1. CPU-only torch, installed from the CPU wheel index. The default PyPI torch
#    wheel pulls ~2.5 GB of CUDA libraries that this service never uses. The
#    index URL is what keeps the image at a few hundred MB instead of several GB.
#
# 2. Two stages. The builder carries compilers and wheel caches; the runtime
#    copies only a built virtualenv. Nothing that can compile code ships.
#
# 3. Non-root, with a fixed uid. A service that has never run as root cannot
#    accidentally write outside its volume, and the uid being fixed means volume
#    permissions are reproducible rather than dependent on the host.

FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# libgomp1 is required by the torch and onnxruntime CPU wheels at import time.
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

# Torch from the CPU index first, so the resolver does not reach for the CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4" \
    && pip install .


FROM python:3.12-slim-bookworm AS runtime

# libgomp1 backs OpenMP, which both torch and onnxruntime need at import.
# curl is only here for the healthcheck.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin mlserve

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Deterministic numerics: one thread each, so timings in the benchmark and
    # latency in production are not functions of how many cores the host has.
    OMP_NUM_THREADS=1 \
    MLSERVE_TORCH_THREADS=1 \
    MLSERVE_HOST=0.0.0.0 \
    MLSERVE_PORT=8000 \
    MLSERVE_DATABASE_URL=sqlite:////data/mlserve.db \
    MLSERVE_ARTIFACTS=/artifacts \
    # Must be set explicitly. Without it, config.py falls back to <repo>/results,
    # which here is /app/results - a directory created by WORKDIR as root, so the
    # non-root service user cannot write to it. The symptom was `init` exiting 1
    # with PermissionError: '/app/results', and it only appears in the container
    # because on a developer machine the repo directory is writable.
    MLSERVE_RESULTS=/results

WORKDIR /app
COPY --chown=mlserve:mlserve pyproject.toml README.md tasks.py ./
COPY --chown=mlserve:mlserve src ./src
COPY --chown=mlserve:mlserve configs ./configs

# Volumes are declared and pre-owned by the service user, so a bind mount from the
# host does not immediately break with a permission error.
#
# /textfile matters for a subtler reason: a named volume is initialised from the
# image's directory at that path, including its ownership. If the path does not
# exist in the image, Docker creates a root-owned volume and the non-root service
# user cannot write to it - which is exactly what happened to the drift job's
# Prometheus textfile output.
RUN mkdir -p /data /artifacts /results /textfile \
    && chown -R mlserve:mlserve /data /artifacts /results /textfile

# Fail the build rather than the first run if any of these is not writable by the
# service user. Cheap, and it catches the whole class of "the code writes somewhere
# the user cannot" defects at build time instead of at 2am.
RUN su mlserve -s /bin/sh -c 'for d in /data /artifacts /results /textfile; do test -w "$d" || { echo "$d is not writable by mlserve"; exit 1; }; done'

USER mlserve
EXPOSE 8000

# No HEALTHCHECK here, deliberately. This image is used by five different roles
# (init, api, drift, drift-publisher, backup) and only one of them serves HTTP, so
# an image-level healthcheck is wrong by construction: it made every non-API
# container report unhealthy and `docker compose up --wait` never returned.
# Healthchecks belong in docker-compose.yml, per service, where they can check
# what that service actually does.

# Default command is the API; the other roles override it.
CMD ["python", "-m", "mlserve.server"]
