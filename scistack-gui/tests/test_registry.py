"""
Tests for scistack_gui.registry — function and variable-class registry.
"""

import types
import pytest

from scidb import BaseVariable
from scistack_gui import registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_module(**attrs) -> types.ModuleType:
    """Create a throwaway module with the given attributes."""
    mod = types.ModuleType("test_pipeline_module")
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# Variable class only used in this test file
class RegistryTestVar(BaseVariable):
    pass


# ---------------------------------------------------------------------------
# register_module
# ---------------------------------------------------------------------------

class TestRegisterModule:
    def test_registers_top_level_callables(self):
        def my_fn(x):
            return x

        mod = _make_module(my_fn=my_fn)
        registry.register_module(mod)
        assert "my_fn" in registry._functions
        assert registry._functions["my_fn"] is my_fn

    def test_skips_private_callables(self):
        def _private(x):
            return x

        mod = _make_module(_private=_private)
        registry.register_module(mod)
        assert "_private" not in registry._functions

    def test_skips_classes(self):
        class MyClass:
            pass

        mod = _make_module(MyClass=MyClass)
        registry.register_module(mod)
        assert "MyClass" not in registry._functions

    def test_multiple_functions(self):
        def fn_a(x): return x
        def fn_b(x): return x

        mod = _make_module(fn_a=fn_a, fn_b=fn_b)
        registry.register_module(mod)
        assert "fn_a" in registry._functions
        assert "fn_b" in registry._functions

    def test_register_twice_overwrites(self):
        def fn_v1(x): return 1
        def fn_v2(x): return 2

        registry.register_module(_make_module(my_fn=fn_v1))
        registry.register_module(_make_module(my_fn=fn_v2))
        assert registry._functions["my_fn"] is fn_v2


# ---------------------------------------------------------------------------
# get_function
# ---------------------------------------------------------------------------

class TestGetFunction:
    def test_returns_registered_function(self):
        def compute(x): return x
        registry._functions["compute"] = compute
        assert registry.get_function("compute") is compute

    def test_raises_key_error_for_unknown_function(self):
        with pytest.raises(KeyError, match="not found in registry"):
            registry.get_function("does_not_exist")

    def test_error_message_includes_function_name(self):
        with pytest.raises(KeyError) as exc_info:
            registry.get_function("missing_fn")
        assert "missing_fn" in str(exc_info.value)


# ---------------------------------------------------------------------------
# get_variable_class
# ---------------------------------------------------------------------------

class TestGetVariableClass:
    def test_returns_registered_variable_class(self):
        # RegistryTestVar is defined at module level — auto-registered on import
        cls = registry.get_variable_class("RegistryTestVar")
        assert cls is RegistryTestVar

    def test_raises_key_error_for_unknown_class(self):
        with pytest.raises(KeyError, match="not found"):
            registry.get_variable_class("NonExistentVarClass")

    def test_error_message_includes_class_name(self):
        with pytest.raises(KeyError) as exc_info:
            registry.get_variable_class("GhostClass")
        assert "GhostClass" in str(exc_info.value)
