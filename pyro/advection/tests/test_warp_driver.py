"""Check the optional backend through the existing Pyro driver and host data API."""
import importlib
import os

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

if os.environ.get("PYRO_REQUIRE_CUDA") == "1":
    importlib.import_module("warp")
else:
    pytest.importorskip("warp")

from pyro import Pyro
from pyro.mesh import patch
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


@pytest.mark.parametrize("periodic_axis", [None, "x", "y"])
@pytest.mark.parametrize("limiter", [0, 1, 2])
def test_outflow_driver_and_boundary_changes(device, periodic_axis, limiter):
    settings = {f"mesh.{axis}{side}boundary": "periodic" if axis == periodic_axis else "outflow"
                for axis in ("x", "y") for side in ("l", "r")}
    settings.update({"advection.limiter": limiter, "advection.u": -0.6,
                     "advection.v": 0.8, "driver.fix_dt": 0.01})
    reference = simulation("numpy", device, **settings)
    actual = simulation("warp", device, **settings)
    retained = actual.sim.cc_data.get_var("density")
    original_backend = actual.sim._warp  # pylint: disable=protected-access
    resumed = None
    for n in range(10):
        if n == 5:
            # Boundary metadata can change without replacing the mesh object.
            for p in (reference, actual, resumed):
                bc = p.sim.cc_data.BCs["density"]
                bc.xlb = bc.xrb = bc.ylb = bc.yrb = "periodic"
                p.sim.cc_data.get_var("density").v()[1:3, 2:4] = 1.7
        reference.single_step()
        actual.single_step()
        if resumed is not None:
            resumed.single_step()
            assert_allclose(resumed.sim.cc_data.get_var("density"), retained, rtol=0, atol=0)
        assert_allclose(retained, reference.sim.cc_data.get_var("density"), rtol=2e-13, atol=2e-13)
        assert actual.sim.n == reference.sim.n
        assert actual.sim.cc_data.t == reference.sim.cc_data.t
        if n == 4:
            actual.sim.write("outflow-continuation")
            loaded = io_pyro.read("outflow-continuation")
            # The HDF5 format stores interior cells; ghosts are filled on use.
            assert_allclose(loaded.cc_data.get_var("density").v(), retained.v(), rtol=0, atol=0)
            assert str(loaded.cc_data.BCs["density"]) == str(actual.sim.cc_data.BCs["density"])
            resumed = simulation("warp", device, **settings)
            resumed.sim.cc_data = loaded.cc_data
            resumed.sim.n = loaded.n
    assert actual.sim._warp is not original_backend  # pylint: disable=protected-access
    assert resumed.sim.n == actual.sim.n
    assert resumed.sim.cc_data.t == actual.sim.cc_data.t


@pytest.mark.parametrize("side", ["xl", "xr", "yl", "yr"])
def test_reject_boundary_value_callbacks(device, side):
    p = simulation("warp", device, **{f"mesh.{axis}{edge}boundary": "outflow"
                   for axis in ("x", "y") for edge in ("l", "r")})
    setattr(p.sim.cc_data.BCs["density"], side + "_value", 1.0)
    with pytest.raises(ValueError, match="boundary value callbacks"):
        p.sim.evolve()


@pytest.mark.parametrize("limiter", [0, 1, 2])
@pytest.mark.parametrize("outflow", [False, True])
@pytest.mark.parametrize("fixed_dt", [-1.0, 0.01])
def test_resident_driver_timesteps_and_edits(device, limiter, outflow, fixed_dt):
    settings = {"advection.limiter": limiter, "driver.fix_dt": fixed_dt,
                "driver.init_tstep_factor": 0.25, "driver.max_dt_change": 1.3,
                "advection.u": -0.6, "advection.v": 0.8}
    if outflow:
        settings.update({"mesh.xlboundary": "outflow", "mesh.xrboundary": "outflow"})
    reference = simulation("numpy", device, **settings)
    actual = simulation("warp", device, **settings, **{"advection.warp_resident": 1})
    # Edits after initialization must supersede the initial device snapshot.
    for p in (reference, actual):
        p.get_var("density").v()[1:3, 2:4] = 1.7
    while not reference.sim.finished():
        reference.single_step()
        actual.single_step()
        assert actual.sim.n == reference.sim.n
        assert actual.sim.cc_data.t == reference.sim.cc_data.t
        assert actual.sim.dt == reference.sim.dt
        if reference.sim.n == 3:
            for p in (reference, actual):
                p.get_var("density").v()[2:4, 3:6] = 1.4
                p.rp.set_param("advection.u", 0.7)
                p.rp.set_param("advection.limiter", 1)
    assert_allclose(actual.get_var("density"), reference.get_var("density"), rtol=2e-13, atol=2e-13)
    assert actual.sim.cc_data.t == 0.137


def test_resident_transfers_and_host_access(device, monkeypatch):
    actual = simulation("warp", device, **{"advection.warp_resident": 1, "driver.fix_dt": 0.01})
    reference = simulation("numpy", device, **{"driver.fix_dt": 0.01})
    core = actual.sim.cc_data.backend
    copies = {"upload": 0, "download": 0}
    original_upload, original_numpy = core.upload, core.numpy

    def upload(density):
        copies["upload"] += 1
        original_upload(density)

    def download():
        copies["download"] += 1
        return original_numpy()

    monkeypatch.setattr(core, "upload", upload)
    monkeypatch.setattr(core, "numpy", download)
    for _ in range(4):
        actual.single_step()
        reference.single_step()
    assert copies == {"upload": 0, "download": 0}
    actual.sim.write("resident-output")
    assert copies == {"upload": 0, "download": 1}
    loaded = io_pyro.read("resident-output")
    assert_allclose(loaded.cc_data.get_var("density").v(), reference.get_var("density").v(), rtol=2e-13, atol=2e-13)
    actual.single_step()
    reference.single_step()
    assert copies == {"upload": 0, "download": 1}
    # Explicit fills followed by host access also materialize current ghosts.
    actual.sim.cc_data.fill_BC_all()
    reference.sim.cc_data.fill_BC_all()
    assert_allclose(actual.sim.cc_data.get_vars(), reference.sim.cc_data.get_vars(), rtol=2e-13, atol=2e-13)
    assert copies == {"upload": 0, "download": 2}
    actual.sim.cc_data.data[5:7, 6:8, 0] = 1.7
    reference.sim.cc_data.data[5:7, 6:8, 0] = 1.7
    actual.single_step()
    reference.single_step()
    assert copies == {"upload": 1, "download": 2}
    assert_allclose(actual.get_var("density"), reference.get_var("density"), rtol=2e-13, atol=2e-13)
    assert copies == {"upload": 1, "download": 3}


def test_resident_output_restart_and_mode_switch(device):
    settings = {"driver.fix_dt": 0.01, "advection.warp_resident": 1,
                "mesh.xlboundary": "outflow", "mesh.xrboundary": "outflow"}
    actual = simulation("warp", device, **settings)
    reference = simulation("numpy", device, **{k: v for k, v in settings.items() if k != "advection.warp_resident"})
    for _ in range(3):
        actual.single_step()
        reference.single_step()
    actual.sim.write("resident-checkpoint")
    loaded = io_pyro.read("resident-checkpoint")
    assert loaded.n == 3
    resumed = simulation("warp", device, **settings)
    resumed.sim.cc_data = loaded.cc_data
    resumed.sim.n = loaded.n
    for _ in range(3):
        actual.single_step()
        reference.single_step()
        resumed.single_step()
    assert_allclose(resumed.get_var("density"), actual.get_var("density"), rtol=0, atol=0)
    # Switch directly from pending device work to host execution.
    for p in (actual, resumed):
        p.single_step()
        p.rp.set_param("advection.warp_resident", 0)
    reference.single_step()
    resumed.rp.set_param("advection.backend", "numpy")
    for p in (actual, resumed, reference):
        p.single_step()
    assert_allclose(actual.get_var("density"), reference.get_var("density"), rtol=2e-13, atol=2e-13)
    assert_allclose(resumed.get_var("density"), reference.get_var("density"), rtol=2e-13, atol=2e-13)


def test_resident_run_sim_and_diagnostics(device):
    actual = simulation("warp", device, **{"advection.warp_resident": 1,
                        "io.force_final_output": 1, "io.basename": "resident_"})
    reference = simulation("numpy", device)
    actual.run_sim()
    reference.run_sim()
    assert actual.sim.n == reference.sim.n
    assert actual.sim.cc_data.min("density") == pytest.approx(reference.sim.cc_data.min("density"), abs=2e-13)
    assert actual.sim.cc_data.max("density") == pytest.approx(reference.sim.cc_data.max("density"), abs=2e-13)
    loaded = io_pyro.read(f"resident_{actual.sim.n:04d}")
    assert_allclose(loaded.cc_data.get_var("density").v(), reference.get_var("density").v(), rtol=2e-13, atol=2e-13)


def test_resident_boundary_fills_after_output(device):
    actual = simulation("warp", device, **{"advection.warp_resident": 1, "driver.fix_dt": 0.01})
    reference = simulation("numpy", device, **{"driver.fix_dt": 0.01})
    for p in (actual, reference):
        p.single_step()
    actual.sim.write("resident-ghosts")
    for p in (actual, reference):
        p.sim.cc_data.fill_BC_all()
    assert_allclose(actual.get_var("density"), reference.get_var("density"), rtol=2e-13, atol=2e-13)


def test_resident_storage_boundaries_and_device_changes(device):
    actual = simulation("warp", device, **{"driver.fix_dt": 0.01})
    reference = simulation("numpy", device, **{"driver.fix_dt": 0.01})
    for p in (actual, reference):
        p.single_step()
    actual.rp.set_param("advection.warp_resident", 1)
    for p in (actual, reference):
        p.single_step()
        cloned = patch.cell_center_data_clone(p.sim.cc_data)
        cloned.t = p.sim.cc_data.t
        p.sim.cc_data = cloned
        cloned.data[5:7, 6:8, 0] = 1.7
        cloned.BCs["density"].xlb = cloned.BCs["density"].xrb = "outflow"
    actual.rp.set_param("advection.warp_device", "cpu")
    for p in (actual, reference):
        p.single_step()
        # Replacing storage without changing cc_data must also upload edits.
        p.sim.cc_data.data = p.sim.cc_data.data.copy()
        p.sim.cc_data.data[6:8, 7:9, 0] = 1.4
        p.single_step()
    assert_allclose(actual.get_var("density"), reference.get_var("density"), rtol=2e-13, atol=2e-13)


def test_resident_scheduled_output_without_uploads(device, monkeypatch):
    settings = {"advection.warp_resident": 1, "driver.fix_dt": 0.01,
                "driver.max_steps": 8, "io.do_io": 1, "io.n_out": 2,
                "io.dt_out": 1.0, "io.basename": "resident_scheduled_"}
    actual = simulation("warp", device, **settings)
    core = actual.sim.cc_data.backend
    copies = {"upload": 0, "download": 0}
    original_upload, original_numpy = core.upload, core.numpy

    def upload(density):
        copies["upload"] += 1
        original_upload(density)

    def download():
        copies["download"] += 1
        return original_numpy()

    monkeypatch.setattr(core, "upload", upload)
    monkeypatch.setattr(core, "numpy", download)
    actual.run_sim()
    assert copies == {"upload": 0, "download": 4}
    reference = simulation("numpy", device, **{"driver.fix_dt": 0.01, "driver.max_steps": 8})
    for _ in range(8):
        reference.single_step()
        if reference.sim.n % 2 == 0:
            loaded = io_pyro.read(f"resident_scheduled_{reference.sim.n:04d}")
            assert loaded.n == reference.sim.n
            assert loaded.cc_data.t == reference.sim.cc_data.t
            assert_allclose(loaded.cc_data.get_var("density").v(), reference.get_var("density").v(),
                            rtol=2e-13, atol=2e-13)


def test_resident_reacquired_views(device):
    actual = simulation("warp", device, **{"advection.warp_resident": 1, "driver.fix_dt": 0.01})
    reference = simulation("numpy", device, **{"driver.fix_dt": 0.01})
    retained = actual.get_var("density")
    snapshot = np.array(retained)
    actual.single_step()
    reference.single_step()
    # Resident mode's opt-in contract: reacquire after stepping.
    assert_allclose(retained.v(), snapshot[4:-4, 4:-4], rtol=0, atol=0)
    fresh = actual.get_var("density")
    assert_allclose(fresh, reference.get_var("density"), rtol=2e-13, atol=2e-13)
    assert_allclose(retained, fresh, rtol=0, atol=0)
