"""ForEachConfig — serializes for_each() computation config into version keys."""

import hashlib
import inspect
import json
from typing import Any, Callable


def _compute_fn_hash(fn: Callable) -> str:
    """SHA-256 of the function's source code, truncated to 16 hex chars.

    Falls back to hashing ``fn.__name__`` if source is unavailable (e.g.
    built-in or compiled functions).  The hash is used downstream by
    check_combo_state to detect whether the function body has changed since
    an output record was saved.
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = getattr(fn, "__name__", repr(fn))
    return hashlib.sha256(src.encode()).hexdigest()[:16]


class ForEachConfig:
    """Serializes for_each() computation config into version keys.

    Captures the parts of a for_each() call that affect the computation's
    identity but are not part of the schema metadata: the function, loadable
    inputs (which variable types / Fixed wrappers are used), where= filter,
    and other behavioral flags.

    These keys are merged into save_metadata so that changing the config
    (e.g. switching smoothing=0.2 to smoothing=0.3, or adding a where= filter)
    creates a new version_keys group rather than silently overwriting existing
    results.
    """

    def __init__(
        self,
        fn: Callable,
        inputs: dict[str, Any],
        where=None,
        distribute: bool = False,
        as_table=None,
    ):
        self.fn = fn
        self.inputs = inputs
        self.where = where
        self.distribute = distribute
        self.as_table = as_table

    def to_version_keys(self) -> dict:
        """Return dict of config keys to merge into save_metadata."""
        keys = {}
        keys["__fn"] = getattr(self.fn, "__name__", repr(self.fn))
        keys["__fn_hash"] = _compute_fn_hash(self.fn)
        inputs_key = self._serialize_inputs()
        if inputs_key != "{}":
            keys["__inputs"] = inputs_key
        direct = self._get_direct_constants()
        if direct:
            keys["__constants"] = json.dumps(direct, sort_keys=True)
        if self.where is not None:
            keys["__where"] = self.where.to_key()
        if self.distribute:
            keys["__distribute"] = True
        if self.as_table:
            if isinstance(self.as_table, list):
                keys["__as_table"] = sorted(self.as_table)
            elif self.as_table is True:
                keys["__as_table"] = True
        return keys

    def _get_direct_constants(self) -> dict:
        """Return scalar constant inputs (non-loadable values)."""
        from .foreach import _is_loadable
        return {k: v for k, v in self.inputs.items() if not _is_loadable(v)}

    def _serialize_inputs(self) -> str:
        """Serialize loadable inputs to a canonical JSON string.

        Only includes loadable inputs (variable types, Fixed, ColumnSelection,
        Merge) — constants are already included in save_metadata directly.
        """
        from .foreach import _is_loadable

        result = {}
        for name in sorted(self.inputs):
            spec = self.inputs[name]
            if _is_loadable(spec):
                if hasattr(spec, "to_key"):
                    result[name] = spec.to_key()
                elif isinstance(spec, type):
                    result[name] = spec.__name__
                else:
                    result[name] = repr(spec)
        return json.dumps(result, sort_keys=True)
