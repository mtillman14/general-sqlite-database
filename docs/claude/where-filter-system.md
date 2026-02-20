# `where=` Filter System

## Purpose

Users frequently need to load one variable conditioned on the value of another. Example: load `StepLength` only where `Side == "L"`. The `where=` parameter provides a composable, Pythonic filter system that requires no SQL knowledge for common cases.

## Syntax

```python
# Equality filter on a whole-variable value
StepLength.load_all(where=Side == "L")

# Comparison filter
StepLength.load_all(where=Speed > 1.2)

# Compound AND / OR
StepLength.load_all(where=(Side == "L") & (Speed > 1.2))
StepLength.load_all(where=(Side == "L") | (Side == "R"))

# NOT
StepLength.load_all(where=~(Side == "L"))

# Column-level filter (tabular variable)
StepLength.load_all(where=GaitData["Side"] == "L")

# Set membership
StepLength.load_all(where=GaitData["Side"].isin(["L", "R"]))

# Raw SQL escape hatch (applied to the target variable's data table)
StepLength.load_all(where=raw_sql('"value" > 0.70'))

# Works with load() too
StepLength.load(subject=1, where=Side == "L")
```

## How It Works

Filters are resolved **before** data is fetched, at the schema_id level:

1. `load()` / `load_all()` calls `db.load_all()` / `db.load()`.
2. `_find_record()` returns the full set of matching `_record_metadata` rows.
3. If `where` is provided, `where.resolve(db, target_class, table_name)` is called.
4. `resolve()` returns `allowed_schema_ids: set[int]`.
5. The records DataFrame is filtered: `records = records[records["schema_id"].isin(allowed_schema_ids)]`.
6. Bulk data loading proceeds as normal over the filtered records.

This is efficient — no unnecessary data is loaded from DuckDB.

## Filter Class Hierarchy

All classes are in `src/scidb/filters.py`.

| Class | Created by | Queries |
|-------|-----------|---------|
| `VariableFilter` | `Side == "L"` (metaclass) | Filter variable's `"value"` column |
| `ColumnFilter` | `GaitData["Side"] == "L"` (ColumnSelection) | Filter variable's named column |
| `InFilter` | `.isin([...])` | Filter variable's column via SQL `IN` |
| `CompoundFilter` | `f1 & f2` or `f1 \| f2` | Set intersection / union of schema_ids |
| `NotFilter` | `~f` | Complement: all target schema_ids minus inner set |
| `RawFilter` | `raw_sql("...")` | Raw SQL applied to the **target** variable's data table |

### resolve() contract

Every filter implements:
```python
def resolve(self, db, target_variable_class, target_table_name) -> set[int]:
    ...
```

Returns the set of `schema_id` integers that pass the filter. `CompoundFilter` combines two sets; `NotFilter` subtracts from the full target schema_id set.

### Latest-version semantics for filter variables

`VariableFilter`, `ColumnFilter`, and `InFilter` all query the filter variable using "latest version per parameter set" semantics (a `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY version_id DESC)` CTE). This mirrors exactly how `load_all(version_id="latest")` works for the target variable — no special casing.

## Metaclass: `VariableMeta`

Defined in `src/scidb/variable.py` and applied to `BaseVariable`.

Overrides `__eq__`, `__ne__`, `__lt__`, `__le__`, `__gt__`, `__ge__` at the **class** level so that `Side == "L"` (where `Side` is a class, not an instance) produces a `VariableFilter`.

Two important preservation rules:
- **Class-to-class equality is preserved**: `Side == Side` returns `True` (standard `type.__eq__` is called when `other` is a `type`).
- **Hashing is preserved**: `__hash__ = type.__hash__` is explicitly set. Without this, overriding `__eq__` on a metaclass would make classes unhashable, breaking dict/set use.

## ColumnSelection operators

`ColumnSelection` (in `scirun-lib/src/scirun/column_selection.py`) gained comparison operators and `isin()`. Each returns a `ColumnFilter` using `self.columns[0]` as the column name. Multi-column selections only use the first column for filtering (consistent with single-column typical usage).

`__hash__` was also added to `ColumnSelection` since `__eq__` was overridden.

## Schema-Level Validation

### Filter at same or coarser level — OK
- **Same level** (e.g., both at `trial`): direct schema_id set match.
- **Coarser level** (e.g., filter at `subject`, target at `trial`): the filter's matching `schema_id`s are **expanded** to all finer-level `schema_id`s in the target that share the same coarse key values. This is a natural hierarchical filter.

### Filter at finer level — Error
Raises `ValueError`:
```
Filter variable 'Side' is stored at schema level 'trial' which is finer than
target 'StepLength' at level 'subject'. Filters must be at the same or coarser
level than the target.
```

### Coverage validation
Every schema location in the target must have a corresponding filter value. If incomplete coverage is detected, raises:
```
Filter variable 'Side' is missing data at 2 schema locations that 'StepLength'
has data for. Ensure the filter variable covers all target locations.
```

### Level detection
Level is inferred by inspecting `_schema` rows for each variable's schema_ids. The deepest non-null column in `dataset_schema_keys` order is the variable's level. This is the same logic used by `_infer_schema_level()` in `DatabaseManager`.

## RawFilter special behavior

`RawFilter` (created by `raw_sql()`) applies the SQL fragment to the **target** variable's own data table (not to the filter variable). This is the only filter type that works this way. It uses a latest-version CTE over the target and injects the raw SQL into the WHERE clause. DuckDB errors propagate wrapped as `ValueError("Invalid where= SQL: ...")`.

## Error messages

| Scenario | Error |
|----------|-------|
| Filter variable has no saved data | `"Filter variable 'X' is not registered. Save data to it first."` |
| Filter at finer level than target | `"Filter variable 'X' is stored at schema level 'y' which is finer than target 'Z' at level 'w'. Filters must be at the same or coarser level than the target."` |
| Filter missing coverage | `"Filter variable 'X' is missing data at N schema locations that 'Z' has data for. Ensure the filter variable covers all target locations."` |
| Raw SQL syntax error | `"Invalid where= SQL: {duckdb_error}"` |

## Key Files

| File | Role |
|------|------|
| `src/scidb/filters.py` | All filter classes + `raw_sql()` factory |
| `src/scidb/variable.py` | `VariableMeta` metaclass + `where=` in `load()`, `load_all()` |
| `src/scidb/__init__.py` | Exports `raw_sql` |
| `src/scidb/database.py` | `where=` in `DatabaseManager.load()` and `load_all()` — calls `resolve()` and filters the records DataFrame |
| `scirun-lib/src/scirun/column_selection.py` | Comparison operators on `ColumnSelection` |
| `tests/test_filters.py` | Unit tests (no DB required) |
| `tests/test_where.py` | Integration tests with real DuckDB |

## What is NOT implemented (deferred)

- **MATLAB wrapper**: MATLAB operator overloading for `where=` is a follow-up. The Python layer is complete and tested.
- **Version-specific filter**: The filter always uses "latest version" of the filter variable. There is no mechanism to specify a particular version of the filter variable.
- **Cross-database filters**: The filter variable must be in the same database as the target.
