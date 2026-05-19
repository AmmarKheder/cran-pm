# CRAN-PM — reproducible CPU/CUDA inference image.
#
# Build:   docker build -f docker/Dockerfile -t cran-pm .
# Run:     docker run --rm cran-pm cranpm --help
# Published: ghcr.io/ammarkheder/cran-pm:<version> (built in CI on tag)
#
# This image is intended for inference and for reproducing the figures
# of the paper. Training on AMD MI250X / LUMI uses the ROCm stack via
# Singularity and is documented separately in the repository README.

FROM python:3.11-slim

LABEL org.opencontainers.image.title="CRAN-PM" \
      org.opencontainers.image.description="Cross-Resolution Attention \
Network for high-resolution PM2.5 forecasting over Europe" \
      org.opencontainers.image.source="https://github.com/AmmarKheder/cran-pm" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Minimal system libs needed by numpy/scipy/netCDF wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# CPU PyTorch wheel + the package itself.
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install .

# Non-root runtime user.
RUN useradd -m runner
USER runner

ENTRYPOINT ["cranpm"]
CMD ["--help"]
