# Device-resident advection measurements

Measured on 2026-10-05 in the pinned Docker environment: Intel(R) Core(TM) i7-14700F, NVIDIA GeForce RTX 4070 Ti SUPER, Warp 1.18.0, and Nsight Systems 2026.3.2.

The uninstrumented benchmark uses periodic smooth float64 density, limiter 2, CFL 0.8, five warmup steps, and five trials of 50 steps. Both modes execute the advection solver through its existing stepping interface. Compatibility mode includes per-step uploads, downloads, and host boundary filling. Resident mode includes the solver and host-mirror bookkeeping, keeps density on the device, and synchronizes once at the end of the timed batch.

Compilation, allocation, trial resets (including their transfers), timestep selection, diagnostics, output, and correctness checks are excluded. These numbers measure stepping rather than a complete simulation. Every final field matches NumPy exactly and passes the mass check.

| Grid | Compatibility CUDA ms/step | Resident CUDA ms/step | Resident / compatibility speedup | Resident / NumPy speedup |
| --- | ---: | ---: | ---: | ---: |
| 32² | 0.147 | 0.083 | 1.78× | 3.46× |
| 64² | 0.157 | 0.083 | 1.89× | 5.09× |
| 128² | 0.177 | 0.083 | 2.13× | 10.95× |
| 256² | 0.298 | 0.086 | 3.46× | 38.49× |
| 512² | 0.640 | 0.085 | 7.56× | 294.48× |
| 1024² | 2.019 | 0.298 | 6.77× | 357.81× |

At 1024², the five compatibility samples range from 1.977 to 2.997 ms/step; resident samples range from 0.298 to 0.299 ms/step.

Separate Nsight captures bracket 100 warmed `Pyro.single_step()` calls at 1024², including timestep bookkeeping. The compatibility capture contains 100 host-to-device and 100 device-to-host copies, totaling 1,704,038,400 bytes. The resident capture contains no memory copies. Both contain 600 numerical kernel launches, and their final fields match NumPy exactly. Use the uninstrumented medians above for speedups; capture overhead affects wall time.

Reproduce the benchmark and capture:

```bash
tools/warp/benchmark.sh --mode driver --output artifacts/residency-driver-final.json
tools/warp/benchmark.sh --mode resident --output artifacts/residency-resident-final.json
tools/warp/profile.sh residency-final-compat
tools/warp/profile.sh residency-final-resident --resident
```

The local `artifacts/` directory contains raw trial timings, PNG plots, package/device metadata, JSON workload reports, Nsight timelines, SQLite exports, and CUDA summaries. Source hashes below identify the implementation used for these benchmarks; the workload JSON files record the same hashes.

| File | SHA-256 |
| --- | --- |
| `pyro/advection/warp_backend.py` | `55fc0d7fefdb58b8e9f8ab62dc6e79886357b5a0589cced81df159170895e60a` |
| `pyro/advection/simulation.py` | `4e0e29ac48955015225f2f05e3a3520df395245279bce0c0ec74e21bb1db1de7` |
| `pyro/advection/warp_data.py` | `2feaa6ec05fbd58fb69df9ca83205782c7090c5580f52ef3acc2d1db7985290f` |
| `tools/warp/benchmark.py` | `fe61c762f28cd9a603d8e773bd9af6414ce302c76ca597a378a36eb25d8a51f7` |
| `tools/warp/requirements.lock` | `3c7558d3e8c9519558f1eac2c4a0406e85e6a2b466d6bbde1eed3c2da1f5ef52` |
