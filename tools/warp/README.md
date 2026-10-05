# Warp advection development tools

Build the pinned development environment from the pyro2 checkout:

```bash
docker build -f tools/warp/Dockerfile -t pyro-warp:dev .
tools/warp/test.sh cuda
```

CUDA execution requires a compatible NVIDIA driver and NVIDIA Container Toolkit
configured for Docker. `tools/warp/test.sh cpu` permits skipped CUDA cases. CUDA
mode sets `PYRO_REQUIRE_CUDA=1` and fails if Warp or CUDA is unavailable. The script
runs the Python unit suite, optional kernel and driver tests, then the original
stored NumPy advection regression. It does not reset benchmarks. Reports are
written to the local `artifacts/` directory.

The optional package extra is `pip install '.[warp]'`. Default NumPy execution
works without Warp. The backend is configured through the existing runtime
parameters `advection.backend=warp` and `advection.warp_device=cuda:0`; particles
must be disabled and all boundaries periodic.

## Benchmark scopes

```bash
# Device-resident core: ghost fills and numerical kernels, no per-step transfers.
tools/warp/benchmark.sh --mode core --output artifacts/advection-core.json

# Existing simulation API: also include uploads, downloads, and host ghost fills.
tools/warp/benchmark.sh --mode driver --output artifacts/advection-driver.json
```

Both modes compare NumPy, Warp CPU, and CUDA using identical smooth float64
fields, limiter 2, CFL 0.8, and fixed timesteps. The defaults are 32²–1024² grids,
five warmup steps, five trials, and 50 timed steps per trial. State is reset before
each trial, and final density and mass are checked outside the timer. CUDA wall
time includes Python submission and final synchronization. Compilation, initial
allocation, trial resets, timestep selection, output, and validation are excluded.
In driver mode, per-step uploads and downloads are timed.

The benchmark writes all trial timings, package/hardware information, source
hashes, a Markdown table, and a PNG chart. Use distinct output names to retain
snapshots across implementation changes. These measurements cover stepping;
complete simulation performance also includes startup, diagnostics, and I/O.

`io_pyro.read` loads data and metadata but does not construct a restart driver.
Continuation tests restore loaded data into a configured driver with matching
parameters. Output and default CPU behavior use the existing upstream interfaces.
