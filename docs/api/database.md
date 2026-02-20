# Database API

The `DatabaseManager` handles all storage operations. You create it once via `configure_database()` and then access it globally via `get_database()` anywhere in your code.

---

## `configure_database()`

Creates and globally registers the database. Call this once at startup — it connects to both the DuckDB data store and the SQLite lineage store, and enables caching for all `@thunk` functions.

=== "Python"

    ```python
    from scidb import configure_database

    db = configure_database(
        "experiment.duckdb",                      # DuckDB file for data
        dataset_schema_keys=["subject", "session"], # which keys identify dataset location
        pipeline_db_path="pipeline.db",            # SQLite file for lineage
    )
    ```

    With lineage mode:

    ```python
    db = configure_database(
        "experiment.duckdb",
        dataset_schema_keys=["subject", "session"],
        pipeline_db_path="pipeline.db",
        lineage_mode="strict",   # default: raise error on unsaved intermediates
        # lineage_mode="ephemeral"  # allow unsaved intermediates without error
    )
    ```

=== "MATLAB"

    ```matlab
    db = scidb.configure_database( ...
        "experiment.duckdb", ...           % DuckDB file
        ["subject", "session"], ...        % schema keys (string array)
        "pipeline.db");                    % SQLite lineage file
    ```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset_db_path` | `str \| Path` | Path to the DuckDB database file (created if it doesn't exist) |
| `dataset_schema_keys` | `list[str]` | **Required.** Keys that identify dataset location vs. computational variants |
| `pipeline_db_path` | `str \| Path` | **Required.** Path to the SQLite lineage file |
| `lineage_mode` | `str` | `"strict"` (default) or `"ephemeral"` |

**Returns:** `DatabaseManager`

### Understanding `dataset_schema_keys`

Schema keys define the "address" in your dataset — like subject, session, trial. Everything else in your metadata becomes a **version key** that distinguishes computational variants at the same address.

```
save(data, subject=1, session="A", smoothing=0.2)
             ↑          ↑              ↑
        schema key  schema key    version key
```

`load(subject=1, session="A")` returns the *latest* version at that address, regardless of `smoothing`. `load(subject=1, session="A", smoothing=0.2)` returns the specific variant.

---

## `get_database()`

Retrieves the globally configured database. Use this when you need the `DatabaseManager` in a function that doesn't have direct access to it.

=== "Python"

    ```python
    from scidb import get_database

    db = get_database()
    provenance = db.get_provenance(FilteredSignal, subject=1, session="A")
    ```

=== "MATLAB"

    ```matlab
    db = scidb.get_database();
    prov = db.get_provenance(py.scidb.variable.BaseVariable, ...);
    % Note: In MATLAB, prefer instance methods: FilteredSignal().provenance(...)
    ```

**Raises:** `DatabaseNotConfiguredError` if called before `configure_database()`

---

## `get_provenance()`

Returns the immediate lineage of a saved variable — the function that produced it, its inputs, and any constant parameters.

=== "Python"

    ```python
    db = get_database()

    prov = db.get_provenance(FilteredSignal, subject=1, session="A")
    if prov:
        print(prov["function_name"])   # "bandpass_filter"
        print(prov["function_hash"])   # SHA-256 of function bytecode
        print(prov["inputs"])          # list of input descriptors
        print(prov["constants"])       # list of {name, value} dicts
    ```

=== "MATLAB"

    ```matlab
    % Preferred: call provenance() on the variable instance
    prov = FilteredSignal().provenance(subject=1, session="A");
    if ~isempty(prov)
        fprintf("Function: %s\n", prov.function_name);
        fprintf("Hash: %s\n", prov.function_hash);
        disp(prov.inputs);
        disp(prov.constants);
    end
    ```

**Returns (Python):** `dict` with keys `function_name`, `function_hash`, `inputs`, `constants`, or `None` if no lineage exists

**Returns (MATLAB):** struct with same fields, or `[]` if no lineage

---

## `get_pipeline_structure()`

Returns a schema-blind view of the pipeline — how variable types are connected through functions, without reference to specific data instances. Useful for understanding the overall computation graph.

=== "Python"

    ```python
    db = get_database()

    structure = db.get_pipeline_structure()
    for step in structure:
        inputs = step["input_types"]
        fn = step["function_name"]
        output = step["output_type"]
        print(f"{inputs} --[{fn}]--> {output}")

    # ['RawSignal'] --[bandpass_filter]--> FilteredSignal
    # ['FilteredSignal'] --[compute_rms]--> RMSValue
    ```

**Returns:** `list[dict]` with keys `input_types` (list of type names), `function_name`, `output_type`

---

## `get_provenance_by_schema()`

Returns all lineage records at a specific schema location — every computation that produced data there.

=== "Python"

    ```python
    db = get_database()

    records = db.get_provenance_by_schema(subject=1)
    for r in records:
        print(f"{r['function_name']} → {r['output_type']}")
        print(f"  output hash: {r['output_content_hash']}")

    # Narrow to a specific subject + session
    records = db.get_provenance_by_schema(subject=1, session="A")
    ```

---

## `export_to_csv()`

Exports all matching records of a variable type to a single CSV file.

=== "Python"

    ```python
    db = get_database()

    count = db.export_to_csv(
        FilteredSignal,
        "filtered_signals.csv",
        session="A",  # optional filter
    )
    print(f"Exported {count} records")
    ```

**Returns:** `int` — number of records exported

---

## Variable Groups

Variable groups are named collections that help you organize related variable types. Groups are stored in the database and persist across sessions.

=== "Python"

    ```python
    from scidb import get_database

    db = get_database()

    # Add to a group (accepts classes, strings, or a list of either)
    db.add_to_var_group("kinematics", StepLength)
    db.add_to_var_group("kinematics", [StepWidth, StepTime])
    db.add_to_var_group("kinematics", "CadenceValue")   # by name string

    # List all group names
    groups = db.list_var_groups()
    # ["emg", "kinematics", "outcomes"]

    # Get all variable classes in a group (sorted alphabetically)
    var_classes = db.get_var_group("kinematics")
    # [<class 'StepLength'>, <class 'StepTime'>, <class 'StepWidth'>]

    for cls in var_classes:
        for var in cls.load_all(subject=1):
            process(var.data)

    # Remove from a group
    db.remove_from_var_group("kinematics", StepTime)
    db.remove_from_var_group("kinematics", ["StepLength", "StepWidth"])
    ```

=== "MATLAB"

    ```matlab
    % Add to a group
    scidb.add_to_var_group("kinematics", {StepLength(), StepWidth(), StepTime()});
    scidb.add_to_var_group("kinematics", {'StepLength', 'StepWidth'});
    scidb.add_to_var_group("kinematics", ["StepLength", "StepWidth"]);
    scidb.add_to_var_group("kinematics", "CadenceValue");  % single string

    % List all group names (returns cell array of strings)
    groups = scidb.list_var_groups();

    % Get variable instances in a group
    vars = scidb.get_var_group("kinematics");
    % vars is a cell array of BaseVariable instances
    for i = 1:numel(vars)
        results = vars{i}.load_all(subject=1);
        % process each result...
    end

    % Remove from a group
    scidb.remove_from_var_group("kinematics", "StepTime");
    scidb.remove_from_var_group("kinematics", ["StepLength", "StepWidth"]);
    ```

Adding the same variable to the same group twice is a no-op (idempotent).

---

## Exceptions

| Exception | Raised When |
|-----------|------------|
| `DatabaseNotConfiguredError` | `get_database()` called before `configure_database()` |
| `NotRegisteredError` | Loading a variable type that has never been saved |
| `NotFoundError` | No records match the metadata query |
| `ReservedMetadataKeyError` | Using a reserved key (`record_id`, `id`, etc.) in metadata |
| `UnsavedIntermediateError` | Strict lineage mode detects unsaved intermediate `ThunkOutput`s |
