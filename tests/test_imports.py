"""Every module in the package must be importable.

Catches typo'd import paths and missing dependencies before they reach a user.
"""
import importlib
import pkgutil

import pytest

import histo_miner

MODULES = [m.name for m in pkgutil.walk_packages(histo_miner.__path__, "histo_miner.")]


def test_package_exposes_version():
    assert isinstance(histo_miner.__version__, str)


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)