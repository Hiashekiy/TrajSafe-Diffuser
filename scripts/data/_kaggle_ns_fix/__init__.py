"""Fabricated ``dataset`` namespace to load Kaggle 2dpathplanning samples.

The samples were torch.save()'d while the author's local package ``dataset``
(with submodules such as ``dataset.map_sample``) was not uploaded.  torch/pickle
therefore fail with ModuleNotFoundError: No module named 'dataset...'.

This module installs a meta-path finder that fabricates ``dataset`` and any
``dataset.*`` submodule.  Every class lookup returns a generic container whose
attributes are set from the pickle state, so ``map / start / goal / path``
attributes become directly accessible.
"""

import importlib.abc
import importlib.util
import sys
import types

_FABRICATED_PREFIX = ("dataset",)


class _LazyContainer:
    """Generic holder: accepts any constructor args and stores keyword attrs."""

    def __init__(self, *args, **kwargs):
        self._args = args
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"<fabricated dataset object {attrs!r}>"


def _make_module(name):
    module = types.ModuleType(name)

    def __getattr__(attr):
        # Fabricate a class: pickle will instantiate it, then set attributes
        # (map/start/goal/path ...) directly from the serialized state.
        def ctor(*args, **kwargs):
            return _LazyContainer(*args, **kwargs)

        return type(attr, (_LazyContainer,), {"__init__": ctor})

    module.__getattr__ = __getattr__  # PEP 562
    module.__path__ = []  # pretend package so ``dataset.x`` sub-imports resolve
    return module


class _FabLoader(importlib.abc.Loader):
    def create_module(self, spec):
        module = _make_module(spec.name)
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module):
        return None


class _FabFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == _FABRICATED_PREFIX[0] or fullname.startswith(_FABRICATED_PREFIX[0] + "."):
            return importlib.util.spec_from_loader(fullname, _FabLoader())
        return None


def ensure_ns_fix():
    """Register the fabricated ``dataset`` namespace (idempotent)."""
    for finder in sys.meta_path:
        if isinstance(finder, _FabFinder):
            return
    sys.meta_path.insert(0, _FabFinder())
