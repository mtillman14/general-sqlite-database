"""
Function and variable class registry.

Populated at startup when the user passes --module. Gives the backend
access to the actual Python objects needed to reconstruct for_each calls.
"""

import importlib.util
import inspect
import logging
from pathlib import Path
from scidb import BaseVariable

logger = logging.getLogger(__name__)

_functions: dict[str, callable] = {}
_module_path: Path | None = None
_module_name: str = "user_pipeline"


def register_module(module, *, module_path: Path | None = None) -> None:
    """
    Scan a user module for pipeline functions and BaseVariable subclasses.

    Functions: any top-level callable that doesn't start with '_'.
    Variable classes: all BaseVariable subclasses currently in memory
      (they self-register on definition via BaseVariable._all_subclasses).

    If module_path is provided, it is stored so that refresh_module() can
    re-import the file later without restarting the server.
    """
    global _module_path
    if module_path is not None:
        _module_path = module_path

    for name, obj in inspect.getmembers(module, lambda o: callable(o) and not inspect.isclass(o)):
        if not name.startswith('_'):
            _functions[name] = obj
    # BaseVariable subclasses are already in _all_subclasses after import —
    # no extra work needed here.


def refresh_module() -> dict:
    """
    Re-import the user module from disk and re-register all functions.

    Returns a summary dict with the old and new function/variable counts
    so the caller can log what changed.
    """
    if _module_path is None:
        raise RuntimeError(
            "No module was loaded at startup (--module not passed). "
            "Nothing to refresh."
        )

    old_fns = set(_functions.keys())
    old_vars = set(BaseVariable._all_subclasses.keys())

    # Clear the function registry so removed functions don't linger.
    _functions.clear()

    # Re-execute the module file. This will re-define all functions and
    # BaseVariable subclasses (which auto-register via the metaclass).
    spec = importlib.util.spec_from_file_location(_module_name, _module_path)
    user_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_mod)

    # Re-scan for functions.
    for name, obj in inspect.getmembers(user_mod, lambda o: callable(o) and not inspect.isclass(o)):
        if not name.startswith('_'):
            _functions[name] = obj

    new_fns = set(_functions.keys())
    new_vars = set(BaseVariable._all_subclasses.keys())

    added_fns = new_fns - old_fns
    removed_fns = old_fns - new_fns
    added_vars = new_vars - old_vars

    if added_fns:
        logger.info("Refresh: added functions %s", added_fns)
    if removed_fns:
        logger.info("Refresh: removed functions %s", removed_fns)
    if added_vars:
        logger.info("Refresh: added variable classes %s", added_vars)

    return {
        "functions": sorted(new_fns),
        "variables": sorted(new_vars),
        "added_functions": sorted(added_fns),
        "removed_functions": sorted(removed_fns),
        "added_variables": sorted(added_vars),
    }


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
