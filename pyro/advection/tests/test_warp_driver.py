"""Check the optional backend through the existing Pyro driver and host data API."""
import importlib
import os

import h5py
import pytest
from numpy.testing import assert_allclose

if os.environ.get("PYRO_REQUIRE_CUDA") == "1":
    importlib.import_module("warp")
else:
    pytest.importorskip("warp")

from pyro import Pyro
from pyro.util import io_pyro

pytestmark = pytest.mark.usefixtures("isolated_outputs")


def simulation(backend, device, **options):
    p = Pyro("advection")
    params = {"mesh.nx": 12, "mesh.ny": 9, "particles.do_particles": 0,
              "advection.backend": backend, "advection.warp_device": device,
              "driver.tmax": 0.137, "driver.max_steps": 1000,
              "io.force_final_output": 0}
    params.update(options)
    p.initialize_problem("smooth", inputs_dict=params)
    return p


@pytest.mark.parametrize("limiter", [0, 1, 2])
@pytest.mark.parametrize("fixed_dt", [-1.0, 0.01])
def test_driver_timesteps_and_host_views(device, limiter, fixed_dt):
    settings = {"advection.limiter": limiter, "driver.fix_dt": fixed_dt,
                "driver.init_tstep_factor": 0.25, "driver.max_dt_change": 1.3}
    reference = simulation("numpy", device, **settings)
    actual = simulation("warp", device, **settings)
    # A retained NumPy view must continue to reflect every evolved step.
    retained = actual.sim.cc_data.get_var("density")
    while not reference.sim.finished():
        reference.single_step()
        actual.single_step()
        assert actual.sim.n == reference.sim.n
        assert actual.sim.cc_data.t == reference.sim.cc_data.t
        assert actual.sim.dt == reference.sim.dt
        assert_allclose(retained, reference.sim.cc_data.get_var("density"), rtol=2e-13, atol=2e-13)
    assert actual.sim.cc_data.t == 0.137
    assert actual.sim.finished()


def test_host_edits_and_parameter_changes(device):
    reference = simulation("numpy", device)
    actual = simulation("warp", device)
    for p in (reference, actual):
        p.single_step()
        p.sim.cc_data.get_var("density").v()[2:4, 3:6] = 1.7
        p.rp.set_param("advection.u", -0.6)
        p.rp.set_param("advection.v", 0.0)
        p.rp.set_param("advection.limiter", 1)
        p.single_step()
    assert_allclose(actual.sim.cc_data.get_var("density"), reference.sim.cc_data.get_var("density"),
                    rtol=2e-13, atol=2e-13)


def test_output_and_save_load_continuation(device):
    reference = simulation("numpy", device, **{"driver.fix_dt": 0.01})
    actual = simulation("warp", device, **{"driver.fix_dt": 0.01})
    for _ in range(3):
        reference.single_step()
        actual.single_step()
    actual.sim.write("warp-checkpoint")
    with h5py.File("warp-checkpoint.h5") as f:
        assert f.attrs["nsteps"] == 3
        assert f.attrs["time"] == actual.sim.cc_data.t
        assert f["runtime parameters"].attrs["advection.backend"] == "warp"
    loaded = io_pyro.read("warp-checkpoint")
    assert loaded.n == 3
    assert_allclose(loaded.cc_data.get_var("density").v(), reference.sim.cc_data.get_var("density").v(),
                    rtol=2e-13, atol=2e-13)
    # io_pyro.read is a data reader, not a driver restart API. Restore that
    # data into a configured driver and continue with identical parameters.
    resumed = simulation("warp", device, **{"driver.fix_dt": 0.01})
    resumed.sim.cc_data = loaded.cc_data
    resumed.sim.n = loaded.n
    while not reference.sim.finished():
        reference.single_step()
        actual.single_step()
        resumed.single_step()
    assert resumed.sim.n == reference.sim.n
    assert resumed.sim.cc_data.t == reference.sim.cc_data.t
    assert_allclose(resumed.sim.cc_data.get_var("density").v(), reference.sim.cc_data.get_var("density").v(),
                    rtol=2e-13, atol=2e-13)
    assert_allclose(resumed.sim.cc_data.get_var("density").v(), actual.sim.cc_data.get_var("density").v(),
                    rtol=0, atol=0)


def test_run_sim_final_output(device):
    p = simulation("warp", device, **{"driver.fix_dt": 0.01,
                   "io.force_final_output": 1, "io.basename": "gpu_"})
    p.run_sim()
    loaded = io_pyro.read(f"gpu_{p.sim.n:04d}")
    assert loaded.n == p.sim.n
    assert loaded.cc_data.t == p.sim.cc_data.t
    assert_allclose(loaded.cc_data.get_var("density").v(), p.sim.cc_data.get_var("density").v(), rtol=0, atol=0)
