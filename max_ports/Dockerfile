# MAX + FastHTML API for Railway (CPU).
# Weights are NOT baked in — download from Hugging Face at boot
# (MATGRAM_HF_REPO + MATGRAM_AUTO_LOAD=1).

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIXI_HOME=/opt/pixi \
    PATH=/opt/pixi/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MATGRAM_DEVICE=cpu \
    MATGRAM_AUTO_LOAD=1 \
    MATGRAM_MAX_PORT=8000 \
    MATGRAM_STARTUP_TIMEOUT=600

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://pixi.sh/install.sh | bash

WORKDIR /app

COPY pixi.toml pixi.lock ./
RUN pixi install --locked

COPY mat_gram01 ./mat_gram01
COPY serve_api ./serve_api
COPY scripts ./scripts
COPY model_assets ./model_assets

EXPOSE 8080

# Railway injects PORT; FastHTML reads it via serve_api.__main__.
CMD ["pixi", "run", "api"]
