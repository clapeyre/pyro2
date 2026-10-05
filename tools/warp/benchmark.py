"""Benchmark identical float64 advection steps, including periodic ghost fills.

Compilation, initialization, state resets, and validation are untimed.
Core/resident-mode transfers are untimed; driver-mode transfers are timed.
Resident mode measures driver stepping with density kept on device.
CUDA wall time includes Python submission and the final device synchronization.
Driver mode also includes per-step uploads and downloads for host API compatibility.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from numpy.testing import assert_allclose

from pyro import Pyro
from pyro.advection.warp_backend import Advection


def summary(samples):
    return {"median_ms": statistics.median(samples), "min_ms": min(samples),
            "max_ms": max(samples), "samples_ms": samples}


def benchmark_size(n, args):
    p = Pyro("advection")
    p.initialize_problem("smooth", inputs_dict={"mesh.nx": n, "mesh.ny": n,
        "particles.do_particles": 0, "advection.limiter": args.limiter,
        "io.force_final_output": 0})
    s = p.sim
    a = s.cc_data.get_var("density")
    g = s.cc_data.grid
    initial = np.array(a, copy=True)
    initial_mass = initial[4:-4, 4:-4].sum()
    dt = args.cfl * min(g.dx, g.dy)  # u = v = 1 in inputs.smooth
    s.dt = dt

    def upstream_step():
        s.cc_data.fill_BC_all()
        s.evolve()

    def reset_upstream():
        a[:] = initial
        s.cc_data.t = 0.0
        s.n = 0

    for _ in range(args.warmup):
        upstream_step()
    baseline_samples = []
    reference = None
    for _ in range(args.repeats):
        reset_upstream()
        start = time.perf_counter()
        for _ in range(args.steps):
            upstream_step()
        baseline_samples.append((time.perf_counter() - start) * 1000 / args.steps)
        if reference is None:
            reference = np.array(a.v(), copy=True)
        else:
            assert_allclose(a.v(), reference, rtol=0, atol=0)
    assert_allclose(reference.sum(), initial_mass, rtol=3e-14, atol=3e-14)
    row = {"nx": n, "ny": n, "dt": dt, "upstream": summary(baseline_samples), "warp": {}}

    for device_name in args.devices:
        if args.mode in ("driver", "resident"):
            driver = Pyro("advection")
            driver.initialize_problem("smooth", inputs_dict={"mesh.nx": n, "mesh.ny": n,
                "particles.do_particles": 0, "advection.limiter": args.limiter,
                "advection.backend": "warp", "advection.warp_device": device_name,
                "advection.warp_resident": int(args.mode == "resident"),
                "io.force_final_output": 0})
            driver.sim.dt = dt

            def step(driver=driver):
                driver.sim.cc_data.fill_BC_all()
                driver.sim.evolve()

            def reset(driver=driver):
                driver.sim.cc_data.get_var("density")[:] = initial
                if args.mode == "resident":
                    # Trial reset transfers are excluded, as in core mode.
                    driver.sim.cc_data.begin_step(driver.sim.cc_data.backend)
                driver.sim.cc_data.t = 0.0
                driver.sim.n = 0

            def result_array(driver=driver):
                return driver.sim.cc_data.get_var("density").v()
        else:
            w = Advection(initial, nx=n, ny=n, dx=g.dx, dy=g.dy,
                                  limiter=args.limiter, device=device_name)
            snapshot = wp.array(initial, dtype=wp.float64, device=device_name)

            def step(w=w):
                w.step(dt)

            def reset(w=w, snapshot=snapshot):
                wp.copy(w.a, snapshot)

            def result_array(w=w):
                return w.numpy()[4:-4, 4:-4]
        for _ in range(args.warmup):
            step()
        wp.synchronize_device(device_name)
        samples = []
        max_error = 0.0
        max_mass_change = 0.0
        for _ in range(args.repeats):
            reset()
            wp.synchronize_device(device_name)
            start = time.perf_counter()
            for _ in range(args.steps):
                step()
            wp.synchronize_device(device_name)
            samples.append((time.perf_counter() - start) * 1000 / args.steps)
            result = result_array()
            assert_allclose(result, reference, rtol=2e-13, atol=2e-13)
            assert_allclose(result.sum(), initial_mass, rtol=3e-14, atol=3e-14)
            max_error = max(max_error, float(np.max(np.abs(result - reference))))
            max_mass_change = max(max_mass_change, float(abs(result.sum()-initial_mass)/abs(initial_mass)))
        measured = summary(samples)
        measured.update(speedup_vs_upstream=row["upstream"]["median_ms"] / measured["median_ms"],
                        max_abs_error=max_error, max_relative_mass_change=max_mass_change,
                        correctness_passed=True)
        row["warp"][device_name] = measured
    if "cpu" in row["warp"]:
        for name, measured in row["warp"].items():
            if name != "cpu":
                measured["speedup_vs_warp_cpu"] = row["warp"]["cpu"]["median_ms"] / measured["median_ms"]
    return row


def cpu_model():
    info = Path("/proc/cpuinfo")
    if info.exists():
        for line in info.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def markdown(report):
    args = report["workload"]
    lines = ["# Advection stepping benchmark", "",
        f"Mode: {args['mode']}. Driver mode includes host/device transfers and host ghost fills; "
        "core and resident modes keep density device-resident. "
        "Resident mode includes the solver/host-mirror bookkeeping.", "", f"Run: {report['started_at_utc']}.", "",
        f"Float64, periodic smooth problem, limiter {args['limiter']}, CFL {args['cfl']}; "
        f"{args['warmup']} warmup steps, {args['steps']} timed steps per trial, {args['repeats']} trials.", "",
        f"CPU: {report['environment']['cpu_model']}. GPUs: " +
        ", ".join(d['name'] for d in report['environment']['devices'] if d['is_cuda']) + ".", "",
        "Median wall time per step includes periodic ghost fills and, on CUDA, Python submission "
        "and final synchronization. Initialization, compilation, resets, core-mode transfers, timestep "
        "selection, output, and validation are excluded (driver-mode state transfers are included). "
        "Each trial starts from the same field "
        "and checks the final density and mass. CUDA graphs are not used.", "",
        "| Grid | pyro2 ms/step | Warp CPU ms/step | CUDA ms/step | CUDA / pyro2 speedup | CUDA / Warp CPU speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["results"]:
        cpu = row["warp"].get("cpu")
        cuda = next((v for k, v in row["warp"].items() if k.startswith("cuda")), None)

        def fmt(value):
            return f"{value:.3f}" if value is not None else "—"
        lines.append(f"| {row['nx']}² | {row['upstream']['median_ms']:.3f} | "
            f"{fmt(cpu['median_ms'] if cpu else None)} | {fmt(cuda['median_ms'] if cuda else None)} | "
            f"{fmt(cuda['speedup_vs_upstream'] if cuda else None)}× | "
            f"{fmt(cuda.get('speedup_vs_warp_cpu') if cuda else None)}× |")
    error = max(v['max_abs_error'] for row in report['results'] for v in row['warp'].values())
    lines += ["", f"All measured trials passed correctness checks; maximum absolute density difference: {error:.3g}.", "",
        "Raw trial timings, min/max spread, dependency versions, device information, workload "
        "parameters, and code hashes are in the companion JSON. This measures the current "
        "stepping API; complete simulations also include startup, diagnostics, visualization, and I/O.", ""]
    return "\n".join(lines)


def plot(report, output):
    plt.switch_backend("Agg")
    rows = report["results"]
    sizes = [r["nx"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    series = [("pyro2", [r["upstream"] for r in rows])]
    for device in report["workload"]["devices"]:
        series.append(("Warp " + device, [r["warp"][device] for r in rows]))
    for label, samples in series:
        med = np.array([r["median_ms"] for r in samples])
        low = np.array([r["min_ms"] for r in samples])
        high = np.array([r["max_ms"] for r in samples])
        axes[0].errorbar(sizes, med, yerr=[med-low, high-med], marker="o", capsize=3, label=label)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Milliseconds per step (median, min–max)")
    axes[0].set_title("Stepping time")
    for device in report["workload"]["devices"]:
        values = [r["warp"][device]["speedup_vs_upstream"] for r in rows]
        axes[1].plot(sizes, values, marker="o", label="Warp " + device + " / pyro2")
    axes[1].axhline(1, color="gray", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Speedup relative to upstream pyro2")
    axes[1].set_title("Relative performance")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes, labels=[f"{n}²" for n in sizes])
        ax.set_xlabel("Grid size")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Float64 periodic CTU advection • {report['workload']['mode']} mode • warmed kernels")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["core", "driver", "resident"], default="core")
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024])
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--limiter", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--cfl", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark-advection.json"))
    args = parser.parse_args()
    if min(args.sizes) < 4 or min(args.steps, args.warmup, args.repeats) < 1 or not 0 < args.cfl <= 1:
        parser.error("sizes >= 4, positive step/trial counts, and 0 < CFL <= 1 required")
    # Resolve output before using a temporary directory for pyro's inputs.auto.
    output = args.output.resolve()
    root = Path(__file__).resolve().parents[2]
    wp.init()
    devices = [wp.get_device(name) for name in args.devices]  # Fail rather than silently omit CUDA.
    report = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
        "workload": {k: v for k, v in vars(args).items() if k != "output"},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
            "cpu_model": cpu_model(), "cpu_count": os.cpu_count(),
            "cuda_driver_version": wp.get_cuda_driver_version(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "numba", "warp-lang", "pyro-hydro")},
            "devices": [{"alias": str(d), "name": d.name, "is_cuda": d.is_cuda,
                         "arch": d.arch if d.is_cuda else None} for d in devices]},
        "upstream_base": "adaf5c59b045664bfd46c68385100241b6196b16",
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root / "pyro/advection/warp_backend.py", root / "pyro/advection/simulation.py",
                      root / "pyro/advection/warp_data.py",
                      Path(__file__).resolve(), root / "tools/warp/requirements.lock")},
        "results": []}
    cwd = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="pyro-benchmark-") as tmp:
            os.chdir(tmp)
            for n in args.sizes:
                row = benchmark_size(n, args)
                report["results"].append(row)
                print(f"{n}x{n}: upstream {row['upstream']['median_ms']:.4f} ms/step; " + "; ".join(
                    f"{name} {v['median_ms']:.4f} ms/step ({v['speedup_vs_upstream']:.2f}x)"
                    for name, v in row['warp'].items()), flush=True)
    finally:
        os.chdir(cwd)
    # Publish only a complete report: a failed correctness check raises before this point.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    output.with_suffix(".md").write_text(markdown(report))
    plot(report, output.with_suffix(".png"))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
