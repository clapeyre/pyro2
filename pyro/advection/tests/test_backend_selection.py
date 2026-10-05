"""Backend validation and default operation do not require Warp installation."""
import subprocess
import sys

import pytest

from pyro import Pyro


@pytest.mark.parametrize("options,message", [
    ({"advection.backend": "invalid"}, "advection.backend"),
    ({"advection.backend": "warp", "particles.do_particles": 1}, "particles"),
    ({"advection.backend": "warp", "mesh.xlboundary": "outflow"}, "periodic"),
    ({"advection.backend": "warp", "mesh.grid_type": "SphericalPolar"}, "Cartesian"),
])
def test_reject_unsupported_configuration(tmp_path, monkeypatch, options, message):
    monkeypatch.chdir(tmp_path)
    p = Pyro("advection")
    with pytest.raises(ValueError, match=message):
        params = {"particles.do_particles": 0}
        params.update(options)
        p.initialize_problem("smooth", inputs_dict=params)


def test_numpy_does_not_import_warp(tmp_path):
    code = '''
import importlib.abc
import sys
class BlockWarp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "warp" or fullname.startswith("warp."):
            raise ModuleNotFoundError("Warp deliberately unavailable", name="warp")
sys.meta_path.insert(0, BlockWarp())
from pyro import Pyro
p = Pyro("advection")
p.initialize_problem("test", inputs_dict={"driver.max_steps": 3, "mesh.nx": 8, "mesh.ny": 8,
                                         "io.force_final_output": 0})
p.run_sim()
assert p.sim.n == 3
assert "warp" not in sys.modules
try:
    p.initialize_problem("smooth", inputs_dict={"advection.backend": "warp", "particles.do_particles": 0})
except ImportError as exc:
    assert "pip install" in str(exc)
else:
    raise AssertionError("Missing Warp must produce an actionable error")
'''
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)
