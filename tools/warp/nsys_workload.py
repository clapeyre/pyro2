"""Capture warmed Pyro.single_step calls with CUDA's profiler API.

Initialization, compilation, output, and final correctness checks are outside
capture. Run without nsys to obtain uninstrumented wall time for the same region.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import warp as wp
from numpy.testing import assert_allclose

from pyro import Pyro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resident", action="store_true")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.size < 4 or min(args.steps, args.warmup) < 1:
        parser.error("size >= 4 and positive step counts required")
    output = args.output.resolve()
    root = Path(__file__).resolve().parents[2]
    started_at = datetime.now(timezone.utc).isoformat()
    settings = {"mesh.nx": args.size, "mesh.ny": args.size, "particles.do_particles": 0,
                "advection.backend": "warp", "advection.warp_device": "cuda:0",
                "driver.fix_dt": 0.8 / args.size,
                "driver.tmax": (args.steps + args.warmup) * 0.8 / args.size + 1.0,
                "driver.max_steps": args.steps + args.warmup, "io.force_final_output": 0}
    if args.resident:
        settings["advection.warp_resident"] = 1
    with tempfile.TemporaryDirectory(prefix="pyro-profile-") as tmp:
        # Pyro writes inputs.auto in the working directory.
        cwd = Path.cwd()
        try:
            os.chdir(tmp)
            p = Pyro("advection")
            p.initialize_problem("smooth", inputs_dict=settings)
            for _ in range(args.warmup):
                p.single_step()
            wp.synchronize_device("cuda:0")
            with wp.ScopedCudaProfiler():
                start = time.perf_counter()
                for _ in range(args.steps):
                    p.single_step()
                wp.synchronize_device("cuda:0")
                elapsed = time.perf_counter() - start
            result = np.array(p.get_var("density").v())
            settings["advection.backend"] = "numpy"
            settings.pop("advection.warp_resident", None)
            reference = Pyro("advection")
            reference.initialize_problem("smooth", inputs_dict=settings)
            for _ in range(args.warmup + args.steps):
                reference.single_step()
            expected = reference.get_var("density").v()
            assert_allclose(result, expected, rtol=2e-13, atol=2e-13)
        finally:
            os.chdir(cwd)
    report = {"started_at_utc": started_at,
              "packages": {name: importlib.metadata.version(name) for name in ("numpy", "warp-lang", "pyro-hydro")},
              "device": wp.get_device("cuda:0").name,
              "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (root / "pyro/advection/simulation.py",
                                          root / "pyro/advection/warp_backend.py",
                                          root / "pyro/advection/warp_data.py", Path(__file__).resolve())},
              "resident": args.resident, "size": args.size, "steps": args.steps,
              "warmup": args.warmup, "elapsed_s": elapsed,
              "ms_per_step": 1000 * elapsed / args.steps,
              "max_abs_error": float(np.max(np.abs(result - expected))),
              "correctness_passed": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
