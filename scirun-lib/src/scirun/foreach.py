"""Batch execution utilities for running functions over metadata combinations."""

from itertools import product
from typing import Any, Callable

from .column_selection import ColumnSelection
from .fixed import Fixed
from .merge import Merge


def for_each(
    fn: Callable,
    inputs: dict[str, Any],
    outputs: list[type],
    dry_run: bool = False,
    save: bool = True,
    pass_metadata: bool | None = None,
    as_table: list[str] | bool | None = None,
    db=None,
    distribute: bool = False,
    where=None,
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
        distribute: If True, split each output (vector/table) by element/row
                    and save each piece at the schema level immediately
                    below the deepest iterated key, using 1-based indexing.
                    For example, with schema [subject, trial, cycle] and
                    iteration at the trial level, distribute=True saves
                    each element/row as a separate cycle (1, 2, 3, ...).
                    No lineage is tracked for distributed saves.
        where: Optional Filter to apply when loading each input variable.
               Combos where an input has no data after filtering are skipped.
               Example: where=Side == "R"  (only process combos where Side is R)
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
    # Resolve empty lists to all distinct values from the database
    needs_resolve = [k for k, v in metadata_iterables.items()
                     if isinstance(v, list) and len(v) == 0]
    if needs_resolve:
        resolved_db = db
        if resolved_db is None:
            try:
                from scidb.database import get_database
                resolved_db = get_database()
            except Exception:
                raise ValueError(
                    f"Empty list [] was passed for {needs_resolve}, which means "
                    f"'use all levels', but no database is available. Either pass "
                    f"db= to for_each or call configure_database() first."
                )
        for key in needs_resolve:
            values = resolved_db.distinct_schema_values(key)
            if not values:
                print(f"[warn] no values found for '{key}' in database, 0 iterations")
            metadata_iterables[key] = values

    # Validate distribute parameter and resolve target key
    distribute_key = None
    if distribute:
        _dist_db = db
        if _dist_db is None:
            try:
                from scidb.database import get_database
                _dist_db = get_database()
            except Exception:
                raise ValueError(
                    "distribute=True requires access to dataset_schema_keys, "
                    "but no database is available. Either pass db= to for_each or "
                    "call configure_database() first."
                )
        schema_keys = _dist_db.dataset_schema_keys

        iter_keys_in_schema = [k for k in schema_keys if k in metadata_iterables]
        if not iter_keys_in_schema:
            raise ValueError(
                "distribute=True requires at least one metadata_iterable "
                "that is a schema key."
            )
        deepest_iterated = iter_keys_in_schema[-1]
        deepest_idx = schema_keys.index(deepest_iterated)

        if deepest_idx + 1 >= len(schema_keys):
            raise ValueError(
                f"distribute=True but '{deepest_iterated}' is the deepest schema key. "
                f"There is no lower level to distribute to. "
                f"Schema order: {schema_keys}"
            )
        distribute_key = schema_keys[deepest_idx + 1]

    # Separate loadable inputs from constants
    loadable_inputs = {}
    constant_inputs = {}
    for param_name, var_spec in inputs.items():
        if _is_loadable(var_spec):
            loadable_inputs[param_name] = var_spec
        else:
            constant_inputs[param_name] = var_spec

    # Check distribute doesn't conflict with a constant input name
    if distribute_key is not None and distribute_key in constant_inputs:
        raise ValueError(
            f"distribute target '{distribute_key}' conflicts with a constant input named '{distribute_key}'."
        )

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
        if distribute_key is not None:
            print(f"[dry-run] distribute: '{distribute_key}' (split outputs by element/row, 1-based)")
        print()

    completed = 0
    skipped = 0

    for values in product(*value_lists):
        metadata = dict(zip(keys, values))
        metadata_str = ", ".join(f"{k}={v}" for k, v in metadata.items())

        if dry_run:
            _print_dry_run_iteration(inputs, outputs, metadata, constant_inputs, should_pass_metadata, distribute_key)
            completed += 1
            continue

        # Load inputs (only loadable ones, not constants)
        loaded_inputs = {}
        load_failed = False

        for param_name, var_spec in loadable_inputs.items():
            # Handle Merge: load each constituent and combine into DataFrame
            if isinstance(var_spec, Merge):
                try:
                    loaded_inputs[param_name] = _load_and_merge(
                        var_spec, metadata, param_name, db
                    )
                except Exception as e:
                    print(f"[skip] {metadata_str}: failed to load {param_name} ({var_spec.__name__}): {e}")
                    load_failed = True
                    break
                continue

            # Guard against Fixed wrapping Merge
            if isinstance(var_spec, Fixed) and isinstance(var_spec.var_type, Merge):
                raise TypeError(
                    "Fixed cannot wrap a Merge. Use Fixed on individual "
                    "constituents inside the Merge instead: "
                    "Merge(Fixed(VarA, ...), VarB)"
                )

            # Resolve var_type, load_metadata, and column_selection from the spec
            var_type, load_metadata, column_selection = _resolve_var_spec(
                var_spec, metadata
            )

            try:
                db_kwargs = {"db": db} if db is not None else {}
                where_kwargs = {"where": where} if where is not None else {}
                loaded_inputs[param_name] = var_type.load(**db_kwargs, **load_metadata, **where_kwargs)
            except Exception as e:
                var_name = getattr(var_type, '__name__', type(var_type).__name__)
                print(f"[skip] {metadata_str}: failed to load {param_name} ({var_name}): {e}")
                load_failed = True
                break

            # Handle as_table conversion and/or column selection
            is_multi = isinstance(loaded_inputs[param_name], list)
            wants_table = param_name in as_table_set and is_multi

            if column_selection is not None and wants_table:
                # Both active: filter each variable's data BEFORE building
                # the table so metadata columns are preserved
                _apply_column_selection_to_vars(
                    loaded_inputs[param_name], column_selection, param_name
                )
                loaded_inputs[param_name] = _multi_result_to_dataframe(
                    loaded_inputs[param_name], var_type
                )
            elif wants_table:
                loaded_inputs[param_name] = _multi_result_to_dataframe(
                    loaded_inputs[param_name], var_type
                )
            elif column_selection is not None:
                loaded_inputs[param_name] = _apply_column_selection(
                    loaded_inputs[param_name], column_selection, param_name
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

            if distribute_key is not None:
                # Distribute mode: split each output and save pieces individually
                for output_type, output_value in zip(outputs, result):
                    raw_value = _unwrap_for_distribute(output_value)
                    try:
                        pieces = _split_for_distribute(raw_value)
                    except TypeError as e:
                        print(f"[error] {metadata_str}: cannot distribute {output_type.__name__}: {e}")
                        continue

                    for i, piece in enumerate(pieces):
                        dist_metadata = {**save_metadata, distribute_key: i + 1}
                        try:
                            output_type.save(piece, **db_kwargs, **dist_metadata)
                            dist_str = ", ".join(f"{k}={v}" for k, v in dist_metadata.items())
                            print(f"[save] {dist_str}: {output_type.__name__}")
                        except Exception as e:
                            dist_str = ", ".join(f"{k}={v}" for k, v in dist_metadata.items())
                            print(f"[error] {dist_str}: failed to save {output_type.__name__}: {e}")
            else:
                # Normal mode: save each output directly
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
    """Check if an input spec is a loadable type (class, Fixed, ColumnSelection, Merge, or has .load())."""
    return isinstance(var_spec, (type, Fixed, ColumnSelection, Merge)) or hasattr(var_spec, 'load')


def _is_thunk(fn: Any) -> bool:
    """Check if fn is a thunk-lib Thunk (without hard dependency)."""
    try:
        from thunk.core import Thunk
        return isinstance(fn, Thunk)
    except ImportError:
        return False


def _unwrap(value: Any) -> Any:
    """Extract raw data from a loaded variable, pass everything else through.

    Skips numpy arrays and DataFrames directly (they don't need unwrapping,
    and numpy arrays have a .data memoryview attribute that must not be accessed).
    """
    # Never unwrap native data types that are already "raw"
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value
    except ImportError:
        pass
    try:
        import pandas as pd
        if isinstance(value, (pd.DataFrame, pd.Series)):
            return value
    except ImportError:
        pass
    if isinstance(value, list):
        return value
    if hasattr(value, 'data'):
        return value.data
    return value


def _unwrap_for_distribute(value: Any) -> Any:
    """Unwrap ThunkOutput/BaseVariable for distribute, but not raw data types.

    Unlike _unwrap, this avoids stripping numpy arrays' .data (memoryview)
    or pandas DataFrames. Only unwraps objects that have .data and are NOT
    themselves a distributable data type.
    """
    # Don't unwrap types that _split_for_distribute can handle directly
    if isinstance(value, list):
        return value
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value
    except ImportError:
        pass
    try:
        import pandas as pd
        if isinstance(value, pd.DataFrame):
            return value
    except ImportError:
        pass
    # For everything else (ThunkOutput, BaseVariable, custom wrappers), unwrap
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

    If all results contain DataFrame data, flattens by merging metadata columns
    with data columns (replicating metadata per row for multi-row data).

    Otherwise, nests data into a single column named after var_type.
    """
    import pandas as pd

    # Check if all data items are DataFrames
    all_dataframes = all(isinstance(var.data, pd.DataFrame) for var in results)

    if all_dataframes:
        # Flatten mode: metadata + data columns merged into one DataFrame
        parts = []
        for var in results:
            data_df = var.data
            meta = dict(var.metadata) if var.metadata else {}
            meta["version_id"] = getattr(var, "version_id", None)
            nr = len(data_df)
            meta_df = pd.DataFrame({k: [v] * nr for k, v in meta.items()})
            parts.append(pd.concat([meta_df.reset_index(drop=True),
                                    data_df.reset_index(drop=True)], axis=1))
        return pd.concat(parts, ignore_index=True)
    else:
        # Non-DataFrame data: nest into a column named after the variable type
        view_name = var_type.view_name() if hasattr(var_type, 'view_name') else var_type.__name__
        rows = []
        for var in results:
            row = dict(var.metadata) if var.metadata else {}
            row["version_id"] = getattr(var, "version_id", None)
            row[view_name] = var.data
            rows.append(row)
        return pd.DataFrame(rows)


def _apply_column_selection(loaded_value: Any, columns: list[str], param_name: str) -> Any:
    """Extract selected columns from loaded data.

    For single column: returns numpy array (the column's .values).
    For multiple columns: returns a DataFrame subset.

    Works on both raw DataFrames and BaseVariable instances with .data as DataFrame.
    """
    import pandas as pd

    # Get the DataFrame — either from .data or directly
    if hasattr(loaded_value, 'data') and isinstance(loaded_value.data, pd.DataFrame):
        df = loaded_value.data
    elif isinstance(loaded_value, pd.DataFrame):
        df = loaded_value
    else:
        data_type = type(getattr(loaded_value, 'data', loaded_value)).__name__
        raise TypeError(
            f"Column selection on '{param_name}' requires DataFrame data, "
            f"but loaded data is {data_type}."
        )

    # Validate column names exist
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"Column(s) {missing} not found in '{param_name}'. "
            f"Available columns: {list(df.columns)}"
        )

    if len(columns) == 1:
        return df[columns[0]].values
    else:
        return df[columns]


def _apply_column_selection_to_vars(variables: list, columns: list[str], param_name: str) -> None:
    """Filter columns on each variable's DataFrame data in-place.

    Used when both column_selection and as_table are active. Keeps data as a
    DataFrame (rather than extracting .values for single columns) so that
    _multi_result_to_dataframe can build the table with metadata + selected columns.
    """
    import pandas as pd

    for var in variables:
        if hasattr(var, 'data') and isinstance(var.data, pd.DataFrame):
            missing = [c for c in columns if c not in var.data.columns]
            if missing:
                raise KeyError(
                    f"Column(s) {missing} not found in '{param_name}'. "
                    f"Available columns: {list(var.data.columns)}"
                )
            var.data = var.data[columns]
        else:
            data_type = type(getattr(var, 'data', var)).__name__
            raise TypeError(
                f"Column selection on '{param_name}' requires DataFrame data, "
                f"but loaded data is {data_type}."
            )


def _format_inputs(inputs: dict[str, Any]) -> str:
    """Format inputs dict for display."""
    parts = []
    for name, var_spec in inputs.items():
        if isinstance(var_spec, Merge):
            parts.append(f"{name}: {var_spec.__name__}")
        elif isinstance(var_spec, Fixed):
            fixed_str = ", ".join(f"{k}={v}" for k, v in var_spec.fixed_metadata.items())
            inner_name = getattr(var_spec.var_type, '__name__', type(var_spec.var_type).__name__)
            parts.append(f"{name}: Fixed({inner_name}, {fixed_str})")
        elif isinstance(var_spec, ColumnSelection):
            parts.append(f"{name}: {var_spec.__name__}")
        elif _is_loadable(var_spec):
            var_name = getattr(var_spec, '__name__', type(var_spec).__name__)
            parts.append(f"{name}: {var_name}")
        else:
            parts.append(f"{name}: {var_spec!r}")
    return "{" + ", ".join(parts) + "}"


def _split_for_distribute(data: Any) -> list[Any]:
    """Split data into elements for distribute-style saving.

    Supports:
    - numpy 1D arrays: split by element
    - numpy 2D arrays: split by row
    - lists: split by element
    - pandas DataFrames: split by row (each row becomes a single-row DataFrame)

    Returns a list of individual pieces.

    Raises:
        TypeError: If data type is not supported for splitting.
    """
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return [data.iloc[[i]] for i in range(len(data))]
    except ImportError:
        pass

    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                return [data[i] for i in range(len(data))]
            elif data.ndim == 2:
                return [data[i, :] for i in range(data.shape[0])]
            else:
                raise TypeError(
                    f"distribute does not support numpy arrays with {data.ndim} dimensions. "
                    f"Only 1D (split by element) and 2D (split by row) are supported."
                )
    except ImportError:
        pass

    if isinstance(data, list):
        return list(data)

    raise TypeError(
        f"distribute does not support type {type(data).__name__}. "
        f"Supported types: numpy 1D/2D array, list, pandas DataFrame."
    )


def _print_dry_run_iteration(
    inputs: dict[str, Any],
    outputs: list[type],
    metadata: dict[str, Any],
    constant_inputs: dict[str, Any],
    pass_metadata: bool = False,
    distribute: str | None = None,
) -> None:
    """Print what would happen for one iteration in dry-run mode."""
    metadata_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
    save_metadata = {**metadata, **constant_inputs}
    save_metadata_str = ", ".join(f"{k}={v}" for k, v in save_metadata.items())
    print(f"[dry-run] {metadata_str}:")

    for param_name, var_spec in inputs.items():
        if isinstance(var_spec, Merge):
            print(f"  merge {param_name}:")
            for i, sub_spec in enumerate(var_spec.var_specs):
                _print_constituent_load(sub_spec, metadata, i)
        elif isinstance(var_spec, Fixed):
            load_metadata = {**metadata, **var_spec.fixed_metadata}
            inner = var_spec.var_type
            if isinstance(inner, ColumnSelection):
                var_name = inner.var_type.__name__
                col_str = ", ".join(inner.columns)
                suffix = f" -> columns: [{col_str}]"
            else:
                var_name = getattr(inner, '__name__', type(inner).__name__)
                suffix = ""
            load_str = ", ".join(f"{k}={v}" for k, v in load_metadata.items())
            print(f"  load {param_name} = {var_name}.load({load_str}){suffix}")
        elif isinstance(var_spec, ColumnSelection):
            load_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
            var_name = var_spec.var_type.__name__
            col_str = ", ".join(var_spec.columns)
            print(f"  load {param_name} = {var_name}.load({load_str}) -> columns: [{col_str}]")
        elif _is_loadable(var_spec):
            load_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
            var_name = getattr(var_spec, '__name__', type(var_spec).__name__)
            print(f"  load {param_name} = {var_name}.load({load_str})")
        else:
            print(f"  constant {param_name} = {var_spec!r}")

    if pass_metadata:
        print(f"  pass metadata: {metadata_str}")

    for output_type in outputs:
        if distribute is not None:
            print(f"  distribute {output_type.__name__} by '{distribute}' (1-based indexing)")
        else:
            print(f"  save {output_type.__name__}.save(..., {save_metadata_str})")


def _print_constituent_load(spec: Any, metadata: dict[str, Any], index: int) -> None:
    """Print a single Merge constituent's load line for dry-run display."""
    if isinstance(spec, Fixed):
        load_metadata = {**metadata, **spec.fixed_metadata}
        inner = spec.var_type
        if isinstance(inner, ColumnSelection):
            var_name = inner.var_type.__name__
            col_str = ", ".join(inner.columns)
            suffix = f" -> columns: [{col_str}]"
        else:
            var_name = getattr(inner, '__name__', type(inner).__name__)
            suffix = ""
        load_str = ", ".join(f"{k}={v}" for k, v in load_metadata.items())
        print(f"    [{index}] {var_name}.load({load_str}){suffix}")
    elif isinstance(spec, ColumnSelection):
        load_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
        var_name = spec.var_type.__name__
        col_str = ", ".join(spec.columns)
        print(f"    [{index}] {var_name}.load({load_str}) -> columns: [{col_str}]")
    else:
        load_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
        var_name = getattr(spec, '__name__', type(spec).__name__)
        print(f"    [{index}] {var_name}.load({load_str})")


def _resolve_var_spec(
    var_spec: Any, metadata: dict[str, Any]
) -> tuple[type, dict[str, Any], list[str] | None]:
    """Resolve a var_spec into (var_type, load_metadata, column_selection).

    Handles Fixed, ColumnSelection, Fixed(ColumnSelection), and plain types.

    Returns:
        var_type: The actual class to call .load() on
        load_metadata: The metadata dict to pass to .load()
        column_selection: list of column names, or None
    """
    column_selection = None
    if isinstance(var_spec, Fixed):
        load_metadata = {**metadata, **var_spec.fixed_metadata}
        inner = var_spec.var_type
        if isinstance(inner, ColumnSelection):
            column_selection = inner.columns
            var_type = inner.var_type
        else:
            var_type = inner
    elif isinstance(var_spec, ColumnSelection):
        load_metadata = metadata
        column_selection = var_spec.columns
        var_type = var_spec.var_type
    else:
        load_metadata = metadata
        var_type = var_spec
    return var_type, load_metadata, column_selection


def _load_and_merge(
    merge_spec: Merge,
    metadata: dict[str, Any],
    param_name: str,
    db: Any | None = None,
) -> "pd.DataFrame":
    """Load each constituent of a Merge and combine into a single DataFrame.

    Each constituent must match exactly one record. DataFrames contribute
    all their columns. Arrays and scalars are added as a column named
    after the variable class name.
    """
    import pandas as pd

    parts = []

    for i, spec in enumerate(merge_spec.var_specs):
        var_type, load_metadata, column_selection = _resolve_var_spec(spec, metadata)
        label = f"{param_name}[{i}]"

        db_kwargs = {"db": db} if db is not None else {}
        loaded = var_type.load(**db_kwargs, **load_metadata)

        # Merge requires exactly one result per constituent
        if isinstance(loaded, list):
            var_name = getattr(var_type, '__name__', type(var_type).__name__)
            raise ValueError(
                f"Merge constituent {var_name} returned {len(loaded)} records "
                f"for {label}, but Merge requires exactly 1 per iteration. "
                f"Use more specific metadata or Fixed() to narrow the match."
            )

        # Apply column selection if applicable
        if column_selection is not None:
            loaded = _apply_column_selection(loaded, column_selection, label)
            # Use the selected column name(s) for naming instead of the var class
            col_name = column_selection[0] if len(column_selection) == 1 else None
        else:
            col_name = None

        var_name = getattr(var_type, '__name__', type(var_type).__name__)
        display_name = col_name if col_name is not None else var_name
        part_df = _constituent_to_dataframe(loaded, display_name, label)
        parts.append((var_name, part_df))

    return _merge_parts(parts, param_name)


def _constituent_to_dataframe(loaded: Any, var_name: str, label: str) -> "pd.DataFrame":
    """Convert a loaded value to a DataFrame for merging.

    DataFrame data: all columns included as-is.
    1D array/list: single column named after the variable class.
    2D array: columns named VarName_0, VarName_1, etc.
    Scalar: single-row, single-column DataFrame.
    """
    import numpy as np
    import pandas as pd

    # Extract raw data from BaseVariable/ThunkOutput wrappers
    raw = loaded
    if hasattr(loaded, 'data') and not isinstance(loaded, (np.ndarray, pd.DataFrame, pd.Series)):
        raw = loaded.data

    if isinstance(raw, pd.DataFrame):
        return raw.reset_index(drop=True)
    elif isinstance(raw, np.ndarray):
        if raw.ndim == 1:
            return pd.DataFrame({var_name: raw})
        elif raw.ndim == 2:
            cols = {f"{var_name}_{j}": raw[:, j] for j in range(raw.shape[1])}
            return pd.DataFrame(cols)
        else:
            raise TypeError(
                f"Merge constituent {label} has {raw.ndim}D array data. "
                f"Only 1D and 2D arrays are supported."
            )
    elif isinstance(raw, (int, float, str, bool)):
        return pd.DataFrame({var_name: [raw]})
    elif isinstance(raw, list):
        return pd.DataFrame({var_name: raw})
    else:
        raise TypeError(
            f"Merge constituent {label} has unsupported data type "
            f"{type(raw).__name__}. Supported: DataFrame, ndarray, scalar, list."
        )


def _merge_parts(
    parts: list[tuple[str, "pd.DataFrame"]], param_name: str
) -> "pd.DataFrame":
    """Merge multiple constituent DataFrames column-wise.

    Validates no column name conflicts and that all multi-row
    constituents have the same row count. Scalars are broadcast.
    """
    import pandas as pd

    if not parts:
        raise ValueError(f"Merge for '{param_name}' has no constituents.")

    # Check column name conflicts
    seen_columns: set[str] = set()
    for var_name, df in parts:
        for col in df.columns:
            if col in seen_columns:
                raise KeyError(
                    f"Column name conflict in Merge for '{param_name}': "
                    f"column '{col}' appears in multiple constituents. "
                    f"Use ColumnSelection to select non-conflicting columns."
                )
            seen_columns.add(col)

    # Determine canonical row count from multi-row parts
    row_counts = [(var_name, len(df)) for var_name, df in parts if len(df) > 1]

    if row_counts:
        unique_counts = set(n for _, n in row_counts)
        if len(unique_counts) > 1:
            detail = ", ".join(f"{name}={n}" for name, n in row_counts)
            raise ValueError(
                f"Cannot merge constituents with different row counts in "
                f"'{param_name}': {detail}. All multi-row constituents must "
                f"have the same number of rows."
            )
        target_len = row_counts[0][1]
    else:
        target_len = 1

    # Broadcast single-row parts to target length, then concat
    expanded = []
    for _var_name, df in parts:
        if len(df) == 1 and target_len > 1:
            df = pd.concat([df] * target_len, ignore_index=True)
        expanded.append(df)

    return pd.concat(expanded, axis=1)
