# mat-gram ESOL HTTP API + MAX Serve (CPU).
#
# Local server: docker compose up -d --build  (bind-mounts model_assets/)
# Weights are NOT baked into the image (.dockerignore) — mount or set
# MATGRAM_HF_REPO for download-at-boot.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIXI_HOME=/opt/pixi \
    PATH=/opt/pixi/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    MATGRAM_DEVICE=cpu \
    MATGRAM_AUTO_LOAD=1 \
    MATGRAM_MAX_PORT=8000 \
    MATGRAM_STARTUP_TIMEOUT=600

# clang + build-essential: MAX JIT-compiles Mojo kernels on import and needs a C linker.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        clang \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://pixi.sh/install.sh | bash

WORKDIR /app

COPY pixi.toml pixi.lock ./
RUN pixi install --locked

COPY mat_gram01 ./mat_gram01
COPY serve_api ./serve_api
COPY scripts ./scripts
COPY model_assets ./model_assets

EXPOSE 8080 8000

# Auto-loads model_assets/smi-ted-esol (local mount or HF) then serves all APIs.
CMD ["pixi", "run", "api-esol"]
