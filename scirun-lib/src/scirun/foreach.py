"""Batch execution utilities for running functions over metadata combinations."""

from itertools import product
from typing import Any, Callable

from .fixed import Fixed


def for_each(
    fn: Callable,
    inputs: dict[str, Any],
    outputs: list[type],
    dry_run: bool = False,
    save: bool = True,
    pass_metadata: bool | None = None,
    as_table: list[str] | bool | None = None,
    db=None,
    **metadata_iterables: list[Any],
) -> None:
    """
    Execute a function for all combinations of metadata, loading inputs
    and saving outputs automatically.

    This function is loosely coupled to the storage layer. It expects:
    - Input types to have a `.load(**metadata)` class method
    - Output types to have a `.save(data, **metadata)` class method

    Inputs can be:
    - Variable types (classes with .load()) — loaded from the database
    - Fixed wrappers — loaded with overridden metadata
    - PathInput instances — resolved to file paths
    - Plain values (constants) — passed directly to the function and
      included in the save metadata as version keys

    Args:
        fn: The function to execute (typically a thunked function, but
            works with any callable)
        inputs: Dict mapping parameter names to variable types,
                Fixed wrappers, or constant values
        outputs: List of variable types for outputs (positional)
        dry_run: If True, only print what would be loaded/saved without
                 actually executing
        save: If True (default), save each function run's output.
              If False, do not save any outputs.
        pass_metadata: If True, pass metadata values as keyword arguments
                      to the function. If None (default), auto-detect based
                      on fn's generates_file attribute.
        as_table: Controls which multi-result loads are converted to
                  pandas DataFrames. Can be:
                  - True: convert all loadable inputs
                  - list of input names: convert only those inputs
                  - False/None/[]: no conversion (default)
                  Only applies when load() returns a list (multiple
                  matches). The DataFrame has columns for metadata keys,
                  version_id, and a data column named after the variable type.
        db: Optional DatabaseManager instance to use for all load/save
            operations instead of the global database.
        **metadata_iterables: Iterables of metadata values to combine

    Example:
        for_each(
            filter_data,
            inputs={"step_length": StepLength, "smoothing": 0.2},
            outputs=[FilteredStepLength],
            subject=subjects,
            session=sessions,
        )

        # Preview what would happen
        for_each(
            filter_data,
            inputs={...},
            outputs=[...],
            dry_run=True,
            subject=subjects,
            ...
        )
    """
    # Separate loadable inputs from constants
    loadable_inputs = {}
    constant_inputs = {}
    for param_name, var_spec in inputs.items():
        if _is_loadable(var_spec):
            loadable_inputs[param_name] = var_spec
        else:
            constant_inputs[param_name] = var_spec

    # Build set of input names to convert to DataFrame
    if as_table is True:
        as_table_set = set(loadable_inputs.keys())
    elif as_table:
        as_table_set = set(as_table)
    else:
        as_table_set = set()

    keys = list(metadata_iterables.keys())
    value_lists = [metadata_iterables[k] for k in keys]

    # Count total iterations for progress
    total = 1
    for v in value_lists:
        total *= len(v)

    fn_name = getattr(fn, "__name__", repr(fn))
    should_pass_metadata = pass_metadata if pass_metadata is not None else getattr(fn, 'generates_file', False)

    if dry_run:
        print(f"[dry-run] for_each({fn_name})")
        print(f"[dry-run] {total} iterations over: {keys}")
        print(f"[dry-run] inputs: {_format_inputs(inputs)}")
        print(f"[dry-run] outputs: {[o.__name__ for o in outputs]}")
        print()

    completed = 0
    skipped = 0

    for values in product(*value_lists):
        metadata = dict(zip(keys, values))
        metadata_str = ", ".join(f"{k}={v}" for k, v in metadata.items())

        if dry_run:
            _print_dry_run_iteration(inputs, outputs, metadata, constant_inputs, should_pass_metadata)
            completed += 1
            continue

        # Load inputs (only loadable ones, not constants)
        loaded_inputs = {}
        load_failed = False

        for param_name, var_spec in loadable_inputs.items():
            if isinstance(var_spec, Fixed):
                load_metadata = {**metadata, **var_spec.fixed_metadata}
                var_type = var_spec.var_type
            else:
                load_metadata = metadata
                var_type = var_spec

            try:
                db_kwargs = {"db": db} if db is not None else {}
                loaded_inputs[param_name] = var_type.load(**db_kwargs, **load_metadata)
            except Exception as e:
                var_name = getattr(var_type, '__name__', type(var_type).__name__)
                print(f"[skip] {metadata_str}: failed to load {param_name} ({var_name}): {e}")
                load_failed = True
                break

            # Convert multi-result to DataFrame if requested
            if param_name in as_table_set and isinstance(loaded_inputs[param_name], list):
                loaded_inputs[param_name] = _multi_result_to_dataframe(
                    loaded_inputs[param_name], var_type
                )

        if load_failed:
            skipped += 1
            continue

        # Call the function
        all_param_names = list(loaded_inputs.keys()) + list(constant_inputs.keys())
        print(f"[run] {metadata_str}: {fn_name}({', '.join(all_param_names)})")

        # For plain functions (not Thunks), unwrap BaseVariable / ThunkOutput
        # inputs to raw .data so existing functions work without modification.
        # Thunks handle their own unwrapping internally.
        if not _is_thunk(fn):
            loaded_inputs = {
                k: v if _is_dataframe(v) else _unwrap(v)
                for k, v in loaded_inputs.items()
            }

        # Merge constants into function arguments
        loaded_inputs.update(constant_inputs)

        try:
            if should_pass_metadata:
                result = fn(**loaded_inputs, **metadata)
            else:
                result = fn(**loaded_inputs)
        except Exception as e:
            print(f"[skip] {metadata_str}: {fn_name} raised: {e}")
            skipped += 1
            continue

        # Normalize single output to tuple
        if not isinstance(result, tuple):
            result = (result,)

        # Save outputs (include constants as version keys in metadata)
        save_metadata = {**metadata, **constant_inputs}
        if save:
            db_kwargs = {"db": db} if db is not None else {}
            for output_type, output_value in zip(outputs, result):
                try:
                    output_type.save(output_value, **db_kwargs, **save_metadata)
                    print(f"[save] {metadata_str}: {output_type.__name__}")
                except Exception as e:
                    print(f"[error] {metadata_str}: failed to save {output_type.__name__}: {e}")

        completed += 1

    # Summary
    print()
    if dry_run:
        print(f"[dry-run] would process {total} iterations")
    else:
        print(f"[done] completed={completed}, skipped={skipped}, total={total}")


def _is_loadable(var_spec: Any) -> bool:
    """Check if an input spec is a loadable type (class, Fixed, or has .load())."""
    return isinstance(var_spec, (type, Fixed)) or hasattr(var_spec, 'load')


def _is_thunk(fn: Any) -> bool:
    """Check if fn is a thunk-lib Thunk (without hard dependency)."""
    try:
        from thunk.core import Thunk
        return isinstance(fn, Thunk)
    except ImportError:
        return False


def _unwrap(value: Any) -> Any:
    """Extract raw data from a loaded variable, pass everything else through."""
    if hasattr(value, 'data'):
        return value.data
    return value


def _is_dataframe(value: Any) -> bool:
    """Check if value is a pandas DataFrame."""
    try:
        import pandas as pd
        return isinstance(value, pd.DataFrame)
    except ImportError:
        return False


def _multi_result_to_dataframe(results: list, var_type: type):
    """Convert a list of loaded variables to a pandas DataFrame.

    Columns: metadata keys + version_id + data column (named after var_type).
    """
    import pandas as pd
    view_name = var_type.view_name() if hasattr(var_type, 'view_name') else var_type.__name__
    rows = []
    for var in results:
        row = dict(var.metadata) if var.metadata else {}
        row["version_id"] = getattr(var, "version_id", None)
        row[view_name] = var.data
        rows.append(row)
    return pd.DataFrame(rows)


def _format_inputs(inputs: dict[str, Any]) -> str:
    """Format inputs dict for display."""
    parts = []
    for name, var_spec in inputs.items():
        if isinstance(var_spec, Fixed):
            fixed_str = ", ".join(f"{k}={v}" for k, v in var_spec.fixed_metadata.items())
            parts.append(f"{name}: Fixed({var_spec.var_type.__name__}, {fixed_str})")
        elif _is_loadable(var_spec):
            var_name = getattr(var_spec, '__name__', type(var_spec).__name__)
            parts.append(f"{name}: {var_name}")
        else:
            parts.append(f"{name}: {var_spec!r}")
    return "{" + ", ".join(parts) + "}"


def _print_dry_run_iteration(
    inputs: dict[str, Any],
    outputs: list[type],
    metadata: dict[str, Any],
    constant_inputs: dict[str, Any],
    pass_metadata: bool = False,
) -> None:
    """Print what would happen for one iteration in dry-run mode."""
    metadata_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
    save_metadata = {**metadata, **constant_inputs}
    save_metadata_str = ", ".join(f"{k}={v}" for k, v in save_metadata.items())
    print(f"[dry-run] {metadata_str}:")

    for param_name, var_spec in inputs.items():
        if isinstance(var_spec, Fixed):
            load_metadata = {**metadata, **var_spec.fixed_metadata}
            var_type = var_spec.var_type
            load_str = ", ".join(f"{k}={v}" for k, v in load_metadata.items())
            var_name = getattr(var_type, '__name__', type(var_type).__name__)
            print(f"  load {param_name} = {var_name}.load({load_str})")
        elif _is_loadable(var_spec):
            load_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
            var_name = getattr(var_spec, '__name__', type(var_spec).__name__)
            print(f"  load {param_name} = {var_name}.load({load_str})")
        else:
            print(f"  constant {param_name} = {var_spec!r}")

    if pass_metadata:
        print(f"  pass metadata: {metadata_str}")

    for output_type in outputs:
        print(f"  save {output_type.__name__}.save(..., {save_metadata_str})")
