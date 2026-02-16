# Variable Groups

## Overview

Variable groups are named collections of variable type names, stored in a `_variable_groups` table in DuckDB. They provide a way to organize related variables without affecting storage or behavior — purely an organizational/bookkeeping tool.

## Architecture

### Storage Layer (SciDuck)

The `_variable_groups` table lives in DuckDB alongside data tables:

```sql
CREATE TABLE _variable_groups (
    group_name VARCHAR NOT NULL,
    variable_name VARCHAR NOT NULL,
    PRIMARY KEY (group_name, variable_name)
)
```

The primary key ensures no duplicate (group, variable) pairs. The low-level methods on `SciDuck` are:
- `add_to_group(group_name, variable_names)` — INSERT with ON CONFLICT DO NOTHING
- `remove_from_group(group_name, variable_names)` — DELETE
- `list_groups()` — SELECT DISTINCT group_name
- `get_group(group_name)` — SELECT variable_name WHERE group_name = ?

All accept `variable_names` as either a single string or a list of strings.

### User-Facing Layer (DatabaseManager)

`DatabaseManager` exposes these as wrapper methods with `var_` prefix for clarity:
- `db.add_to_var_group(group_name, variables)`
- `db.remove_from_var_group(group_name, variables)`
- `db.list_var_groups()`
- `db.get_var_group(group_name)`

`add_to_var_group` and `remove_from_var_group` accept either:
- A `BaseVariable` subclass (e.g. `StepLength`)
- A string variable name (e.g. `"StepLength"`)
- A list/iterable of either (can be mixed)

Resolution is handled by two static methods:
- `_resolve_var_name(v)` — resolves a single element: `str` passthrough, `BaseVariable` subclass via `table_name()`, or fallback to `str(v)` for MATLAB objects
- `_resolve_var_names(variables)` — wraps scalars (`str`, `type`) into a list, then iterates any iterable (Python list, MATLAB cell array, MATLAB string array) and resolves each element

These delegate to `self._duck.add_to_group(...)` etc. after resolving names.

### Naming Convention

The DatabaseManager methods use `var_group` (e.g. `add_to_var_group`) while the underlying SciDuck methods use just `group` (e.g. `add_to_group`). The `var_` prefix was added at the DatabaseManager level so the method names are self-descriptive when called from user code or MATLAB.

## Key Behaviors

- **Idempotent adds**: Adding the same variable to the same group twice is a no-op (ON CONFLICT DO NOTHING).
- **Accepts classes or strings**: `add_to_var_group` and `remove_from_var_group` resolve `BaseVariable` subclasses to their `table_name()` automatically.
- **Persistence**: Groups are stored in the DuckDB file and persist across sessions.
- **Sorted results**: Both `list_var_groups()` and `get_var_group()` return alphabetically sorted lists.
- **`get_var_group` returns classes**: In Python it returns `BaseVariable` subclasses (not strings). In MATLAB it returns a cell array of BaseVariable instances. Raises `NotRegisteredError` if a name in the group has no registered subclass.

## MATLAB Access

MATLAB cannot auto-convert BaseVariable objects to Python, so dedicated MATLAB wrapper functions handle conversion before calling the Python layer. These live in `scidb-matlab/src/scidb_matlab/matlab/+scidb/`:

- `scidb.add_to_var_group(group_name, variables)`
- `scidb.remove_from_var_group(group_name, variables)`
- `scidb.list_var_groups()`
- `scidb.get_var_group(group_name)`

The wrappers use `scidb.internal.resolve_var_names(variables)` to convert MATLAB inputs to a `py.list` of name strings before calling the Python `DatabaseManager` method. It handles:
- Cell array of BaseVariable instances → `class(v)` extracts each name
- Cell array of chars → passes through
- String array → `cellstr()` conversion
- Single string/char/BaseVariable → wrapped into a single-element list

```matlab
scidb.add_to_var_group("kinematics", {StepLength(), StepWidth()});
scidb.add_to_var_group("kinematics", {'StepLength', 'StepWidth'});
scidb.add_to_var_group("kinematics", ["StepLength", "StepWidth"]);

vars = scidb.get_var_group("kinematics");
```

## Test Coverage

Tests are in `sciduck/tests/test_sciduck.py` under `class TestGroups`. They cover:
- Adding multiple variables to a group
- Adding a single string (not a list)
- Removing from a group
- Listing all groups
- Idempotent duplicate adds
