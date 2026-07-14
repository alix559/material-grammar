#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export LD_LIBRARY_PATH="$ROOT/vendor/wgpu-mojo/ffi/lib:${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
RUN=(mojo run -I vendor/wgpu-mojo mojo_manim/src/render.mojo)
if [[ -z "${DISPLAY:-}" ]] && command -v xvfb-run >/dev/null; then
  exec xvfb-run -a "${RUN[@]}"
fi
exec "${RUN[@]}"
