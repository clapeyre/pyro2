#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
case "${1:-cpu}" in
    cpu) gpu_args=(); env_args=() ;;
    cuda) gpu_args=(--gpus all); env_args=(-e PYRO_REQUIRE_CUDA=1) ;;
    *) echo 'Usage: tools/warp/test.sh [cpu|cuda]' >&2; exit 2 ;;
esac
mkdir -p artifacts
docker run --rm "${gpu_args[@]}" "${env_args[@]}" --user "$(id -u):$(id -g)" \
    -e MPLCONFIGDIR=/tmp/matplotlib -e NUMBA_CACHE_DIR=/tmp/numba-cache \
    -v "$PWD:/work" -w /work pyro-warp:dev python -m pytest -q pyro \
    -o cache_dir=/tmp/pytest-cache --junitxml=/work/artifacts/unit.xml
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" -w /tmp \
    -e MPLCONFIGDIR=/tmp/matplotlib -e NUMBA_CACHE_DIR=/tmp/numba-cache \
    pyro-warp:dev python -m pyro.test --solver advection --outfile /work/artifacts/regression.txt
