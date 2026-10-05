import importlib
import itertools
import os
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("PYRO_REQUIRE_CUDA") == "1":
    importlib.import_module("warp")
else:
    pytest.importorskip("warp")
from numpy.testing import assert_allclose

import pyro
from pyro import Pyro
from pyro.advection.advective_fluxes import unsplit_fluxes
from pyro.advection.interface import linear_interface
from pyro.advection.warp_backend import PeriodicAdvection
from pyro.mesh import reconstruction
from pyro.util import io_pyro

pytestmark = pytest.mark.usefixtures("isolated_outputs")

VELOCITIES = list(itertools.product([-0.7, 0.0, 0.9], [-0.6, 0.0, 0.8]))


def reference(nx=12, ny=9, u=0.9, v=0.8, limiter=2, profile="random"):
    p = Pyro("advection")
    p.initialize_problem("smooth", inputs_dict={
        "mesh.nx": nx, "mesh.ny": ny, "mesh.xmax": 1.3, "mesh.ymax": 0.8,
        "particles.do_particles": 0, "advection.u": u, "advection.v": v,
        "advection.limiter": limiter, "io.force_final_output": 0})
    s = p.sim
    a = s.cc_data.get_var("density")
    g = s.cc_data.grid
    if profile == "random":
        a.v()[:] = np.random.default_rng(2026).uniform(0.1, 2, (nx, ny))
    elif profile == "constant":
        a.v()[:] = 1.25
    elif profile == "discontinuous":
        a.v()[:] = (g.x2d.v() < 0.6) & (g.y2d.v() < 0.4)
    elif profile == "sine":
        a.v()[:] = 1 + 0.2 * np.sin(2*np.pi*g.x2d.v()/1.3) * np.cos(2*np.pi*g.y2d.v()/0.8)
    s.cc_data.fill_BC_all()
    return s


def port(s, device):
    g = s.cc_data.grid
    return PeriodicAdvection(s.cc_data.get_var("density"), nx=g.nx, ny=g.ny,
        dx=g.dx, dy=g.dy, u=s.rp.get_param("advection.u"),
        v=s.rp.get_param("advection.v"), limiter=s.rp.get_param("advection.limiter"), device=device)


def timestep(s):
    s.method_compute_timestep()
    # Static velocity also has a finite, useful verification timestep.
    return min(s.dt, 0.03)


@pytest.mark.parametrize("u,v", VELOCITIES)
@pytest.mark.parametrize("limiter", [0, 1, 2])
@pytest.mark.parametrize("profile", ["random", "discontinuous", "constant"])
def test_each_numerical_stage(device, u, v, limiter, profile):
    s = reference(u=u, v=v, limiter=limiter, profile=profile)
    w = port(s, device)
    dt = timestep(s)
    w.prepare(dt)
    a = s.cc_data.get_var("density")
    g = s.cc_data.grid
    assert_allclose(w.a.numpy(), a, rtol=0, atol=0)
    for name, direction in [("sx", 1), ("sy", 2)]:
        assert_allclose(getattr(w, name).numpy(), reconstruction.limit(a, g, direction, limiter), rtol=2e-14, atol=2e-14)
    _, _, ax, ay = linear_interface(a, g, s.rp, dt)
    fx, fy = unsplit_fluxes(s.cc_data, s.rp, dt, "density", linear_interface)
    for name, expected in [("ax", ax), ("ay", ay), ("fx", fx), ("fy", fy)]:
        assert_allclose(getattr(w, name).numpy(), expected, rtol=2e-14, atol=2e-14)
    w.step(dt)
    s.dt = dt
    s.evolve()
    assert_allclose(w.numpy()[4:-4, 4:-4], a.v(), rtol=3e-14, atol=3e-14)


@pytest.mark.parametrize("u,v", VELOCITIES)
@pytest.mark.parametrize("limiter", [0, 1, 2])
def test_40_steps_and_mass(device, u, v, limiter):
    s = reference(u=u, v=v, limiter=limiter, profile="sine")
    w = port(s, device)
    initial_mass = s.cc_data.get_var("density").v().sum()
    dt = timestep(s)
    for _ in range(40):
        s.cc_data.fill_BC_all()
        s.dt = dt
        s.evolve()
        w.step(dt)
    result = w.numpy()[4:-4, 4:-4]
    assert np.isfinite(result).all()
    assert_allclose(result, s.cc_data.get_var("density").v(), rtol=2e-13, atol=2e-13)
    assert_allclose(result.sum(), initial_mass, rtol=3e-14, atol=3e-14)


@pytest.mark.parametrize("limiter", [0, 1, 2])
def test_second_order_convergence(device, limiter):
    errors = []
    for n in (16, 32, 64):
        s = reference(nx=n, ny=n, u=0.7, v=-0.4, limiter=limiter, profile="sine")
        w = port(s, device)
        g = s.cc_data.grid
        end = 0.4 * 1.3 / 0.7
        nsteps = n
        dt = end / nsteps
        for _ in range(nsteps):
            w.step(dt)
        exact = 1 + 0.2*np.sin(2*np.pi*(g.x2d.v()-0.7*end)/1.3)*np.cos(2*np.pi*(g.y2d.v()+0.4*end)/0.8)
        errors.append(np.mean(np.abs(w.numpy()[4:-4, 4:-4]-exact)))
    assert errors[0]/errors[1] > 3.0, errors
    assert errors[1]/errors[2] > 3.0, errors


def test_periodic_corners_and_constant(device):
    s = reference(nx=4, ny=5, profile="constant")
    w = port(s, device)
    for _ in range(10):
        w.step(0.01)
    assert_allclose(w.numpy(), 1.25, rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [{"limiter": 3}, {"dx": 0}, {"u": float("nan")}, {"nx": 3}])
def test_invalid_configuration(kwargs):
    params = {"nx": 8, "ny": 8, "dx": 0.1, "dy": 0.1}
    params.update(kwargs)
    with pytest.raises(ValueError):
        PeriodicAdvection(np.ones((16, 16)), **params)


@pytest.mark.parametrize("dt", [0, -1, float("nan"), 2])
def test_invalid_step(dt):
    w = PeriodicAdvection(np.ones((16, 16)), nx=8, ny=8, dx=0.1, dy=0.1)
    with pytest.raises(ValueError):
        w.step(dt)


def test_stored_smooth_regression(device):
    """Use the checked-in HDF5 oracle, with explicit atol unlike upstream compare."""
    p = Pyro("advection")
    p.initialize_problem("smooth", inputs_dict={"particles.do_particles": 0,
        "io.force_final_output": 0})
    s = p.sim
    s.cc_data.fill_BC_all()
    w = port(s, device)
    while not s.finished():
        s.cc_data.fill_BC_all()
        s.compute_timestep()
        w.step(s.dt)
        s.evolve()
    bench = io_pyro.read(str(Path(pyro.__file__).parent / "advection/tests/smooth_0040"))
    result = w.numpy()[4:-4, 4:-4]
    assert s.n == 40
    assert s.cc_data.t == 1.0
    assert_allclose(result, s.cc_data.get_var("density").v(), rtol=2e-13, atol=2e-13)
    assert_allclose(result, bench.cc_data.get_var("density").v(), rtol=1e-12, atol=1e-13)


def test_poisoned_periodic_ghosts(device):
    s = reference(nx=4, ny=5, profile="random")
    g = s.cc_data.grid
    density = np.array(s.cc_data.get_var("density"))
    density[:4, :] = -999
    density[-4:, :] = -999
    density[:, :4] = -999
    density[:, -4:] = -999
    w = PeriodicAdvection(density, nx=g.nx, ny=g.ny, dx=g.dx, dy=g.dy, device=device)
    w.prepare(0.01)
    assert_allclose(w.a.numpy(), s.cc_data.get_var("density"), rtol=0, atol=0)


def test_mc_axis_advection_bounds(device):
    s = reference(nx=32, ny=8, u=0.9, v=0, limiter=1, profile="discontinuous")
    w = port(s, device)
    initial = s.cc_data.get_var("density").v()
    lo, hi = initial.min(), initial.max()
    dt = timestep(s)
    for _ in range(80):
        w.step(dt)
    result = w.numpy()[4:-4, 4:-4]
    assert result.min() >= lo - 1e-14
    assert result.max() <= hi + 1e-14
