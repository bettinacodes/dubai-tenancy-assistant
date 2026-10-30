"""Every module must at least import (a syntax error slipped into a committed module on Day 8 because
no test imported it)."""
import importlib
import pkgutil

import pytest

import tenancy_rag

MODULES = sorted(m.name for m in pkgutil.walk_packages(tenancy_rag.__path__, "tenancy_rag."))


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)
