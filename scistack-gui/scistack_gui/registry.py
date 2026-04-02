"""
Function and variable class registry.

Populated at startup when the user passes --module. Gives the backend
access to the actual Python objects needed to reconstruct for_each calls.
"""

import inspect
from scidb import BaseVariable

_functions: dict[str, callable] = {}


def register_module(module) -> None:
    """
    Scan a user module for pipeline functions and BaseVariable subclasses.

    Functions: any top-level callable that doesn't start with '_'.
    Variable classes: all BaseVariable subclasses currently in memory
      (they self-register on definition via BaseVariable._all_subclasses).
    """
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith('_'):
            _functions[name] = obj
    # BaseVariable subclasses are already in _all_subclasses after import —
    # no extra work needed here.


def get_function(name: str):
    fn = _functions.get(name)
    if fn is None:
        raise KeyError(
            f"Function '{name}' not found in registry. "
            f"Did you pass --module with the script that defines it?"
        )
    return fn


def get_variable_class(name: str) -> type:
    cls = BaseVariable._all_subclasses.get(name)
    if cls is None:
        raise KeyError(
            f"Variable class '{name}' not found. "
            f"Did you pass --module with the script that defines it?"
        )
    return cls
