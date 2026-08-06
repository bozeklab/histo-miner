"""Every module in the package must be importable.

Catches typo'd import paths and missing dependencies before they reach a user.
"""
import importlib
import pkgutil
import sys

import pytest

import histo_miner

MODULES = [m.name for m in pkgutil.walk_packages(histo_miner.__path__, "histo_miner.")]

# Packages declared in the `wsi` and `ml` extras — never imported at module scope.
OPTIONAL_DEPS = ["imagesize", "openslide", "boruta", "mrmr", "xgboost", "lightgbm"]


def test_package_exposes_version():
    assert isinstance(histo_miner.__version__, str)


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_package_imports_without_optional_dependencies(monkeypatch):
    """Nothing in the `wsi` or `ml` extras may be imported at module scope."""
    for name in OPTIONAL_DEPS:
        monkeypatch.setitem(sys.modules, name, None)
    for name in list(sys.modules):
        if name.startswith("histo_miner"):
            monkeypatch.delitem(sys.modules, name, raising=False)

    for name in MODULES:
        importlib.import_module(name)