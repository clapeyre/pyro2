#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p artifacts
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
    -e MPLCONFIGDIR=/tmp/matplotlib -e NUMBA_CACHE_DIR=/tmp/numba-cache \
    -v "$PWD:/work" -w /work pyro-warp:dev python tools/warp/benchmark.py "$@"
