"""DB-backed for_each wrapper — loads inputs, delegates loop to scifor, saves outputs."""

import json
import warnings
from typing import Any, Callable

import scifor as _scifor
from scifor import for_each as _scifor_for_each
from scifor.pathinput import PathInput

from .colname import ColName
from .column_selection import ColumnSelection
from .fixed import Fixed
from .foreach_config import ForEachConfig
from .merge import Merge


def for_each(
    fn: Callable,
    inputs: dict[str, Any],
    outputs: list[type],
    dry_run: bool = False,
    save: bool = True,
    as_table: list[str] | bool | None = None,
    db=None,
    distribute: bool = False,
    where=None,
    **metadata_iterables: list[Any],
) -> "pd.DataFrame | None":
    """
    Execute a function for all combinations of metadata, loading inputs
    and saving outputs automatically.

    This is the DB-backed wrapper. It:
    1. Resolves empty lists ``[]`` via ``db.distinct_schema_values()``
    2. Pre-filters schema combos via ``db.distinct_schema_combinations()``
    3. Builds ``ForEachConfig`` version keys
    4. Loads all input variables into DataFrames
    5. Converts scidb wrappers → scifor wrappers
    6. Delegates the core loop to ``scifor.for_each``
    7. Saves results from the returned table

    Args:
        fn: The function to execute (plain function handle, no Thunk wrapping).
        inputs: Dict mapping parameter names to variable types, Fixed wrappers,
                Merge wrappers, ColumnSelection wrappers, PathInput, or constants.
        outputs: List of output types/objects with ``.save()``.
        dry_run: If True, only print what would happen without executing.
        save: If True (default), save each function run's output.
        as_table: Controls which inputs are passed as full DataFrames.
        db: Optional database instance.
        distribute: If True, split outputs and save each piece at the schema
                    level below the deepest iterated key.
        where: Optional filter; passed to .load() calls on DB-backed inputs.
        **metadata_iterables: Iterables of metadata values to combine.

    Returns:
        A pandas DataFrame of results, or None when dry_run=True.
    """
    # Resolve empty lists to all distinct values from the database
    needs_resolve = [k for k, v in metadata_iterables.items()
                     if isinstance(v, list) and len(v) == 0]
    resolved_db = None
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

    # Propagate schema keys to scifor so distribute and DataFrame detection work
    _propagate_schema(db, distribute)

    # Build ForEachConfig version keys (DB-specific; not part of scifor)
    config = ForEachConfig(
        fn=fn,
        inputs=inputs,
        where=where,
        distribute=distribute,
        as_table=as_table,
    )
    config_keys = config.to_version_keys()

    # Pre-filter to only schema combinations that actually exist in the database.
    all_combos = None
    if needs_resolve and not _has_pathinput(inputs):
        from scidb.database import _schema_str
        filter_db = resolved_db
        schema_keys_set = set(filter_db.dataset_schema_keys)
        keys = list(metadata_iterables.keys())
        schema_indices = [i for i, k in enumerate(keys) if k in schema_keys_set]
        filter_keys = [keys[i] for i in schema_indices]

        if filter_keys:
            from itertools import product
            value_lists = [metadata_iterables[k] for k in keys]
            raw_combos = list(product(*value_lists))

            existing = filter_db.distinct_schema_combinations(filter_keys)
            existing_set = set(existing)

            filtered = [
                dict(zip(keys, combo))
                for combo in raw_combos
                if tuple(_schema_str(combo[i]) for i in schema_indices) in existing_set
            ]
            removed = len(raw_combos) - len(filtered)
            if removed > 0:
                print(f"[info] filtered {removed} non-existent schema combinations "
                      f"(from {len(raw_combos)} to {len(filtered)})")
            all_combos = filtered

    # Build output_names for scifor
    output_names = [_output_name(o) for o in outputs] if outputs else ["result"]

    # Load all inputs into DataFrames (with __record_id and __branch_params)
    loaded_inputs = _convert_inputs(inputs, db, where)

    # --- Variant tracking: build rid→bp mapping and __rid_{param} discriminator columns ---
    import pandas as pd
    from itertools import product as _iproduct

    rid_to_bp: dict = {}   # {record_id: branch_params_dict}
    rid_keys: list = []    # __rid_{param_name} column names added to this call's schema

    for param_name, data in list(loaded_inputs.items()):
        if not isinstance(data, pd.DataFrame) or "__record_id" not in data.columns:
            continue

        # Build rid→bp from this input's DataFrame
        bp_col = "__branch_params" if "__branch_params" in data.columns else None
        for _, row in data.iterrows():
            rid = row["__record_id"]
            if rid is None:
                continue
            bp_raw = row[bp_col] if bp_col else "{}"
            rid_to_bp[rid] = json.loads(bp_raw or "{}") if isinstance(bp_raw, str) else {}

        # Rename __record_id → __rid_{param_name} so per-param tracking is unambiguous
        rid_col = f"__rid_{param_name}"
        loaded_inputs[param_name] = data.rename(columns={"__record_id": rid_col})
        rid_keys.append(rid_col)

    # Strip __branch_params from all DataFrames (now tracked via rid_to_bp)
    for param_name, data in list(loaded_inputs.items()):
        if isinstance(data, pd.DataFrame) and "__branch_params" in data.columns:
            loaded_inputs[param_name] = data.drop(columns=["__branch_params"])

    # --- Build full combos: base_combos × valid rid-combos per schema location ---
    current_schema_keys = list(_scifor.get_schema() or [])

    base_combos = all_combos
    if base_combos is None:
        keys = list(metadata_iterables.keys())
        value_lists = [metadata_iterables[k] for k in keys]
        base_combos = [dict(zip(keys, combo)) for combo in _iproduct(*value_lists)]

    # For each rid_key, map schema_combo_tuple → [rid_values at that combo]
    rid_per_combo: dict = {}
    for rid_col in rid_keys:
        param_name = rid_col[len("__rid_"):]
        df = loaded_inputs.get(param_name)
        if not isinstance(df, pd.DataFrame) or rid_col not in df.columns:
            continue
        schema_cols_in_df = [k for k in current_schema_keys if k in df.columns]
        mapping: dict = {}
        if schema_cols_in_df:
            for combo_vals, group in df.groupby(schema_cols_in_df, sort=False):
                raw_key = combo_vals if isinstance(combo_vals, tuple) else (combo_vals,)
                key = tuple(str(v) for v in raw_key)
                mapping[key] = group[rid_col].tolist()
        else:
            mapping[()] = df[rid_col].tolist()
        rid_per_combo[rid_col] = mapping

    # Expand each base combo with all valid rid-combos for that schema location
    full_combos: list = []
    for combo in base_combos:
        schema_vals = tuple(str(combo.get(k, "")) for k in current_schema_keys)

        rid_lists: list = []
        rid_col_names: list = []
        valid = True
        for rid_col, mapping in rid_per_combo.items():
            rids = mapping.get(schema_vals, [])
            if not rids:
                valid = False
                break
            rid_lists.append(rids)
            rid_col_names.append(rid_col)

        if not valid:
            continue

        if rid_lists:
            for rid_combo in _iproduct(*rid_lists):
                full_combo = {**combo}
                for rc_name, rc_val in zip(rid_col_names, rid_combo):
                    full_combo[rc_name] = rc_val
                full_combos.append(full_combo)
        else:
            full_combos.append(combo)

    # Temporarily extend scifor's schema to include __rid_* keys so _filter_df_for_combo
    # treats them as schema columns (not data columns), giving single-row filtered DFs.
    if rid_keys:
        _scifor.set_schema(current_schema_keys + rid_keys)

    # Collect all rid values per key so scifor's metadata_iterables are complete
    extended_metadata_iterables = dict(metadata_iterables)
    for rid_col, mapping in rid_per_combo.items():
        all_rids: list = []
        for rids in mapping.values():
            all_rids.extend(rids)
        extended_metadata_iterables[rid_col] = list(dict.fromkeys(all_rids))  # preserve order, dedupe

    # Delegate core loop to scifor
    result_tbl = _scifor_for_each(
        fn,
        loaded_inputs,
        dry_run=dry_run,
        as_table=as_table,
        distribute=distribute,
        output_names=output_names,
        _all_combos=full_combos,
        **extended_metadata_iterables,
    )

    # Restore scifor's schema
    if rid_keys:
        _scifor.set_schema(current_schema_keys)

    if result_tbl is None:
        return None

    # Save results
    if save and outputs and not result_tbl.empty:
        _save_results(result_tbl, outputs, output_names, config_keys, db,
                      rid_to_bp=rid_to_bp, rid_keys=rid_keys)

    return result_tbl


# ---------------------------------------------------------------------------
# Input loading and conversion
# ---------------------------------------------------------------------------

def _convert_inputs(
    inputs: dict[str, Any],
    db: Any | None,
    where: Any | None,
) -> dict[str, Any]:
    """Convert all inputs: load var types into DataFrames, convert wrappers.

    Returns a dict suitable for scifor.for_each (DataFrames + constants).
    """
    result = {}
    for param_name, var_spec in inputs.items():
        if isinstance(var_spec, ColName):
            result[param_name] = _resolve_colname_from_db(var_spec, db)
        elif _is_loadable(var_spec):
            result[param_name] = _load_input(var_spec, db, where)
        else:
            # Constant — pass through unchanged
            result[param_name] = var_spec
    return result


def _resolve_colname_from_db(colname: "ColName", db: Any | None) -> str:
    """Resolve a ColName wrapper to the single data column name string.

    Uses the variable's dtype metadata from the database to determine
    what data columns exist, then subtracts schema keys.
    """
    import json

    var_type = colname.var_type

    # Get the database
    resolved_db = db
    if resolved_db is None:
        try:
            from scidb.database import get_database
            resolved_db = get_database()
        except Exception:
            raise ValueError(
                "ColName requires a database to resolve column names. "
                "Either pass db= to for_each or call configure_database() first."
            )

    var_name = var_type.__name__ if isinstance(var_type, type) else type(var_type).__name__
    schema_keys = list(resolved_db.dataset_schema_keys)

    # Query the _variables table for dtype metadata
    try:
        row = resolved_db._execute(
            "SELECT dtype FROM _variables WHERE variable_name = ?",
            [var_name],
        ).fetchone()
    except Exception:
        row = None

    if row is None:
        # Variable not yet saved — try using view_name for single-column mode
        if hasattr(var_type, 'view_name'):
            return var_type.view_name()
        return var_name

    dtype_meta = json.loads(row[0])
    mode = dtype_meta.get("mode", "single_column")

    if mode == "single_column":
        # Single-column variables always have exactly one data column
        col_names = list(dtype_meta.get("columns", {}).keys())
        if col_names:
            return col_names[0]
        if hasattr(var_type, 'view_name'):
            return var_type.view_name()
        return var_name

    if mode == "dataframe":
        # DataFrame variables: subtract schema keys from df_columns
        df_columns = dtype_meta.get("df_columns", list(dtype_meta.get("columns", {}).keys()))
        data_cols = [c for c in df_columns if c not in schema_keys]
        if len(data_cols) == 1:
            return data_cols[0]
        elif len(data_cols) == 0:
            raise ValueError(
                f"ColName({var_name}): variable has no data columns "
                f"(all columns are schema keys). "
                f"Columns: {df_columns}, schema keys: {schema_keys}"
            )
        else:
            raise ValueError(
                f"ColName({var_name}): variable has {len(data_cols)} "
                f"data columns ({data_cols}), expected exactly 1. "
                f"Schema keys: {schema_keys}"
            )

    if mode == "multi_column":
        raise ValueError(
            f"ColName({var_name}): not supported for dict-type (multi_column) variables. "
            f"ColName only works with single-column or single-data-column DataFrame variables."
        )

    # Unknown mode — fall back to view_name
    if hasattr(var_type, 'view_name'):
        return var_type.view_name()
    return var_name


def _load_input(var_spec: Any, db: Any | None, where: Any | None) -> Any:
    """Load a single input and return a scifor-compatible wrapper or DataFrame."""
    import pandas as pd

    # Already a DataFrame — pass through
    if isinstance(var_spec, pd.DataFrame):
        return var_spec

    # Merge: load each constituent and return scifor.Merge of DataFrames
    if isinstance(var_spec, Merge):
        loaded_tables = []
        for sub_spec in var_spec.var_specs:
            loaded = _load_input(sub_spec, db, where)
            loaded_tables.append(loaded)
        return _scifor.Merge(*loaded_tables)

    # Fixed: load inner, return scifor.Fixed with loaded data
    if isinstance(var_spec, Fixed):
        inner_loaded = _load_input(var_spec.var_type, db, where)
        return _scifor.Fixed(inner_loaded, **var_spec.fixed_metadata)

    # ColumnSelection: load inner var_type, return scifor.ColumnSelection
    if isinstance(var_spec, ColumnSelection):
        loaded_df = _load_var_type_all(var_spec.var_type, db, where)
        return _scifor.ColumnSelection(loaded_df, var_spec.columns)

    # PathInput: load is per-combo, so we can't preload.
    # Convert to a DataFrame with metadata columns and a path column.
    if isinstance(var_spec, PathInput):
        # PathInput is special — it resolves paths per-combo, not from DB.
        # We pass it through as a constant; the wrapped fn will handle it.
        # Actually, we need to handle it differently. For now, return as-is
        # and let the wrapped fn handle path resolution.
        return var_spec

    # Variable type (class with .load()): bulk load all records into a DataFrame
    if isinstance(var_spec, type) or hasattr(var_spec, 'load'):
        return _load_var_type_all(var_spec, db, where)

    # Unknown — return as-is
    return var_spec


def _load_var_type_all(
    var_type: Any,
    db: Any | None,
    where: Any | None,
) -> "pd.DataFrame":
    """Bulk load all records for a variable type into a DataFrame with metadata columns."""
    import pandas as pd

    db_kwargs = {"db": db} if db is not None else {}
    where_kwargs = {"where": where} if where is not None else {}

    # Use load_all (not load) to avoid AmbiguousVersionError when multiple
    # variants exist at the same schema location — we want ALL variants here.
    loaded = list(var_type.load_all(version_id="latest", **db_kwargs, **where_kwargs))

    if not loaded:
        return pd.DataFrame()

    # Determine schema keys to stringify — scifor compares schema cols as strings
    # (metadata_iterables values come from the user as strings, e.g. session="1"),
    # but the database may return typed values (e.g. session=np.int64(1)).
    _schema_keys: set = set()
    _resolved_db = db
    if _resolved_db is None:
        try:
            from scidb.database import get_database
            _resolved_db = get_database()
        except Exception:
            pass
    if _resolved_db is not None and hasattr(_resolved_db, 'dataset_schema_keys'):
        _schema_keys = set(_resolved_db.dataset_schema_keys)

    def _stringify_meta(meta: dict) -> dict:
        """Convert schema key values to strings and drop __ version keys."""
        return {k: str(v) if k in _schema_keys and v is not None else v
                for k, v in meta.items()
                if not k.startswith("__")}

    # Check if data is DataFrames
    first = loaded[0]
    all_have_data = all(hasattr(v, 'data') for v in loaded)

    if all_have_data and isinstance(first.data, pd.DataFrame):
        # DataFrame data: build table with metadata cols + data cols
        parts = []
        for var in loaded:
            data_df = var.data
            meta = _stringify_meta(dict(var.metadata) if hasattr(var, 'metadata') and var.metadata else {})
            meta["__record_id"] = getattr(var, 'record_id', None)
            meta["__branch_params"] = json.dumps(getattr(var, 'branch_params', None) or {})
            nr = len(data_df)
            meta_df = pd.DataFrame({k: [v] * nr for k, v in meta.items()})
            parts.append(pd.concat([meta_df.reset_index(drop=True),
                                    data_df.reset_index(drop=True)], axis=1))
        return pd.concat(parts, ignore_index=True)
    elif all_have_data:
        # Scalar/other data: build table with metadata cols + view_name/class_name col
        view_name = var_type.view_name() if hasattr(var_type, 'view_name') else getattr(var_type, '__name__', type(var_type).__name__)
        rows = []
        for var in loaded:
            row = _stringify_meta(dict(var.metadata) if hasattr(var, 'metadata') and var.metadata else {})
            row[view_name] = var.data
            row["__record_id"] = getattr(var, 'record_id', None)
            row["__branch_params"] = json.dumps(getattr(var, 'branch_params', None) or {})
            rows.append(row)
        return pd.DataFrame(rows)
    else:
        # Raw results without .data attribute — just wrap
        rows = []
        var_name = getattr(var_type, '__name__', type(var_type).__name__)
        for var in loaded:
            rows.append({var_name: var, "__record_id": getattr(var, 'record_id', None), "__branch_params": "{}"})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def _save_results(
    result_tbl: "pd.DataFrame",
    outputs: list[Any],
    output_names: list[str],
    config_keys: dict,
    db: Any | None,
    rid_to_bp: "dict | None" = None,
    rid_keys: "list | None" = None,
) -> None:
    """Save results from the result table to output variable types."""
    import pandas as pd

    db_kwargs = {"db": db} if db is not None else {}

    # Get schema keys for dynamic discriminator detection
    schema_keys_set: set = set()
    if db is not None and hasattr(db, 'dataset_schema_keys'):
        schema_keys_set = set(db.dataset_schema_keys)
    else:
        try:
            schema_keys_set = set(_scifor.get_schema() or [])
        except Exception:
            pass

    # Determine which columns are metadata (not output names)
    meta_cols = [c for c in result_tbl.columns if c not in output_names]

    fn_name = config_keys.get("__fn", "")
    direct_constants = json.loads(config_keys.get("__constants", "{}") or "{}")

    for _, row in result_tbl.iterrows():
        # 1. Collect upstream branch_params via __rid_* columns → rid_to_bp lookup
        merged_bp: dict = {}
        if rid_to_bp and rid_keys:
            for rid_col in rid_keys:
                if rid_col not in row.index:
                    continue
                rid = row[rid_col]
                if rid and rid in rid_to_bp:
                    for k, v in rid_to_bp[rid].items():
                        if k in merged_bp and merged_bp[k] != v:
                            warnings.warn(
                                f"branch_params key '{k}' overwritten: "
                                f"{merged_bp[k]!r} → {v!r}. "
                                f"Use version= for precise selection.",
                                UserWarning, stacklevel=4,
                            )
                        merged_bp[k] = v

        # 2. Add constants namespaced by function name
        for k, v in direct_constants.items():
            merged_bp[f"{fn_name}.{k}"] = v

        # 3. Add dynamic discriminators (non-schema, non-__ meta columns with scalar values)
        _scalar_types = (bool, int, float, str)
        for col in meta_cols:
            if col.startswith("__"):
                continue
            if col in schema_keys_set:
                continue
            val = row[col] if col in row.index else None
            if val is None:
                continue
            if isinstance(val, float) and pd.isna(val):
                continue
            if not isinstance(val, _scalar_types):
                continue  # Skip numpy arrays and other complex types
            if col in merged_bp and merged_bp[col] != val:
                warnings.warn(
                    f"branch_params key '{col}' overwritten: "
                    f"{merged_bp[col]!r} → {val!r}. "
                    f"Use version= for precise selection.",
                    UserWarning, stacklevel=4,
                )
            merged_bp[col] = val

        # Build save metadata: only non-__ cols (schema keys etc.) + config_keys + __branch_params
        # Exclude __rid_* and other internal __ columns from version keys.
        save_metadata = {
            col: row[col] for col in meta_cols if not col.startswith("__")
        }
        save_metadata.update(config_keys)
        save_metadata["__branch_params"] = json.dumps(merged_bp)

        for output_obj, output_name in zip(outputs, output_names):
            if output_name not in row.index:
                continue
            output_value = row[output_name]
            try:
                output_obj.save(output_value, **db_kwargs, **save_metadata)
                meta_str = ", ".join(f"{k}={v}" for k, v in save_metadata.items()
                                     if not k.startswith("__"))
                print(f"[save] {meta_str}: {_output_name(output_obj)}")
            except Exception as e:
                meta_str = ", ".join(f"{k}={v}" for k, v in save_metadata.items()
                                     if not k.startswith("__"))
                print(f"[error] {meta_str}: failed to save {_output_name(output_obj)}: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_loadable(var_spec: Any) -> bool:
    """Check if an input spec is loadable (var type, Fixed, Merge, ColumnSelection, etc.)."""
    try:
        import pandas as pd
        if isinstance(var_spec, pd.DataFrame):
            return True
    except ImportError:
        pass
    return isinstance(var_spec, (type, Fixed, ColumnSelection, Merge, PathInput)) or hasattr(var_spec, 'load')


def _has_pathinput(inputs: dict) -> bool:
    """Check if any input is a PathInput, directly or wrapped in Fixed."""
    for v in inputs.values():
        if isinstance(v, PathInput):
            return True
        if isinstance(v, Fixed) and isinstance(v.var_type, PathInput):
            return True
    return False


def _output_name(output_obj: Any) -> str:
    """Get display name for an output object."""
    if hasattr(output_obj, 'view_name'):
        return output_obj.view_name()
    if isinstance(output_obj, type):
        return output_obj.__name__
    return getattr(output_obj, '__name__', type(output_obj).__name__)


def _propagate_schema(db, distribute: bool) -> None:
    """Propagate dataset_schema_keys from the db into scifor.set_schema()."""
    # If a db was passed explicitly and has schema keys, use them.
    if db is not None and hasattr(db, 'dataset_schema_keys'):
        _scifor.set_schema(list(db.dataset_schema_keys))
        return

    # No explicit db: try the global database.
    _global_db = None
    try:
        from scidb.database import get_database
        _global_db = get_database()
    except Exception:
        pass

    if _global_db is not None and hasattr(_global_db, 'dataset_schema_keys'):
        _scifor.set_schema(list(_global_db.dataset_schema_keys))
    elif distribute:
        raise ValueError(
            "distribute=True requires access to dataset_schema_keys, "
            "but no database is available. Either pass db= to for_each or "
            "call configure_database() first."
        )
