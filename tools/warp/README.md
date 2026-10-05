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
must be disabled. Boundaries may be periodic or outflow, with periodic sides
paired along each axis.

## Benchmark scopes

```bash
# Device-resident core: ghost fills and numerical kernels, no per-step transfers.
tools/warp/benchmark.sh --mode core --output artifacts/advection-core.json

# Existing simulation API: also include uploads, downloads, and host ghost fills.
tools/warp/benchmark.sh --mode driver --output artifacts/advection-driver.json

# Advection driver with lazy host synchronization and device-resident steps.
tools/warp/benchmark.sh --mode resident --output artifacts/advection-resident.json
```

All modes compare NumPy, Warp CPU, and CUDA using identical smooth float64
fields, limiter 2, CFL 0.8, and fixed timesteps. The defaults are 32²–1024² grids,
five warmup steps, five trials, and 50 timed steps per trial. State is reset before
each trial, and final density and mass are checked outside the timer. CUDA wall
time includes Python submission and final synchronization. Compilation, initial
allocation, trial resets, timestep selection, output, and validation are excluded.
In driver mode, per-step uploads and downloads are timed. Resident mode includes
the solver and host-mirror bookkeeping; it resets through the host interface
before each trial and downloads for validation afterward, outside the timer.

The benchmark writes all trial timings, package/hardware information, source
hashes, a Markdown table, and a PNG chart. Use distinct output names to retain
snapshots across implementation changes. These measurements cover stepping;
complete simulation performance also includes startup, diagnostics, and I/O.

`io_pyro.read` loads data and metadata but does not construct a restart driver.
Continuation tests restore loaded data into a configured driver with matching
parameters. Output and default CPU behavior use the existing upstream interfaces.

## Device-resident mode and host views

Select `advection.backend=warp advection.warp_resident=1` to keep density on the
device between driver steps. Host array access synchronizes and makes any edits
authoritative before the next step. Reacquire arrays after stepping; retained
NumPy views can be stale until another host access synchronizes them. The default
`warp_resident=0` preserves the original behavior of retained mutable views.
Output and visualization synchronize through the existing host interface.
The built-in HDF5 writer avoids re-uploading unchanged output data.

## Nsight Systems

With host `nsys` installed, profile 100 warmed `Pyro.single_step()` calls at 1024²:

```bash
tools/warp/profile.sh compat
tools/warp/profile.sh resident --resident
```

The script mounts the host Nsight installation read-only into the Docker
container and captures CUDA activity between Warp's profiler start/stop calls.
It disables CPU sampling and context-switch tracing; elevated profiling
capabilities are not needed. Compilation, warmup, output, and final NumPy
correctness checks are excluded. Artifacts include the `.nsys-rep` timeline,
SQLite export, CUDA API/kernel/memory summaries, and a checked workload JSON.
`nsys_workload.py` can also be run without nsys for an uninstrumented measurement.
See [NVIDIA's profiling documentation](https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/profiling.html)
for Warp's capture API. Use benchmark medians for speedups; profiling adds overhead.

The [device-resident measurements](residency-results.md) include benchmark medians,
Nsight copy/kernel counts, reproducible commands, and implementation hashes.
