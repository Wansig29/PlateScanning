"""Every module must at least import: the UI has no other test that would notice a syntax error."""
import importlib
import os
import pkgutil

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import platescanner  # noqa: E402

MODULES = [m.name for m in pkgutil.walk_packages(platescanner.__path__, "platescanner.")
           if not m.name.endswith("__main__")]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)
