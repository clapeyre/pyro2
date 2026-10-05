#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ $# -lt 1 ]]; then
    echo 'Usage: tools/warp/profile.sh REPORT_NAME [nsys_workload.py arguments]' >&2
    exit 2
fi
report_name=$1
shift
if [[ ! $report_name =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo 'REPORT_NAME must contain only letters, digits, underscores, and hyphens' >&2
    exit 2
fi
nsys_root=$(dirname "$(dirname "$(readlink -f "$(command -v nsys)")")")
mkdir -p artifacts
nsys --version > "artifacts/$report_name.nsys-version.txt"
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
    -e MPLCONFIGDIR=/tmp/matplotlib -e NUMBA_CACHE_DIR=/tmp/numba-cache \
    -e XDG_CACHE_HOME=/tmp/nsight-cache -v "$PWD:/work" -v "$nsys_root:/nsight:ro" -w /work \
    pyro-warp:dev /nsight/target-linux-x64/nsys profile \
    --trace=cuda --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
    --capture-range-end=stop --force-overwrite=true \
    --output="/work/artifacts/$report_name" \
    python tools/warp/nsys_workload.py --output "/work/artifacts/$report_name.json" "$@"
nsys stats --force-export=true \
    --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
    "artifacts/$report_name.nsys-rep" > "artifacts/$report_name.stats.txt"
