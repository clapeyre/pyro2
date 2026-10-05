"""Fixtures for optional Warp tests; set PYRO_REQUIRE_CUDA=1 for GPU validation."""
import importlib
import os

import pytest


@pytest.fixture(scope="session", params=["cpu", "cuda:0"])
def device(request):
    if os.environ.get("PYRO_REQUIRE_CUDA") == "1":
        wp = importlib.import_module("warp")
    else:
        wp = pytest.importorskip("warp")
    wp.init()
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        if os.environ.get("PYRO_REQUIRE_CUDA") == "1":
            pytest.fail("CUDA was required but is unavailable")
        pytest.skip("CUDA unavailable")
    return request.param


@pytest.fixture
def isolated_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
