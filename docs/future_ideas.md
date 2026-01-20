# Future Ideas

This document captures ideas for future enhancements that are too complex to implement immediately but are worth preserving for later consideration.

## Provenance-Based Version Inference (`db.infer_versions()`)

### The Problem

When you first start saving data, you may not be tracking version parameters. Later, you realize that certain upstream parameters (like `upstream=2` vs `upstream=4`) matter for distinguishing computational versions. At this point, you have:

- Old records with `version: {}` (no version info)
- New records with `version: {"upstream": 4}` (explicit version info)

You want to retroactively populate the version info for old records.

### The Solution

Since lineage already captures the full computation graph, we can infer version parameters by examining the lineage chain:

```python
# Examine lineage and backfill version keys
db.infer_versions(
    ProcessedSignal,
    rules={
        # "Look in lineage for calls to 'set_upstream', extract 'arg_1' constant"
        "upstream": {"function": "set_upstream", "extract": "arg_1"}
    }
)
```

This would:
1. Find records with empty `version: {}`
2. Walk their lineage chain
3. Find where `set_upstream()` was called
4. Extract the value that was passed as `arg_1`
5. Update `version: {"upstream": <extracted_value>}`

### Implementation Notes

The lineage record already stores:
- `function_name`: Name of the function
- `function_hash`: Hash of the function bytecode
- `inputs`: List of input descriptors (thunk outputs or variables)
- `constants`: List of constant values passed to the function

For `infer_versions()`, we would:

1. Query records where `json_extract(metadata, '$.version') = '{}'`
2. For each record, get its lineage via `get_full_lineage()`
3. Walk the lineage tree looking for matching function names
4. Extract the specified constant value
5. Update the metadata in both the variable table and `_version_log`

### Example Rules

```python
rules = {
    # Simple: extract arg_1 from set_upstream()
    "upstream": {"function": "set_upstream", "extract": "arg_1"},

    # With fallback: try multiple functions
    "window_size": [
        {"function": "smooth", "extract": "window"},
        {"function": "moving_average", "extract": "window_size"},
    ],

    # From any function matching a pattern
    "filter_order": {"function_pattern": "butter_*", "extract": "order"},
}
```

### Challenges

1. **Ambiguity**: What if the same function appears multiple times in the lineage with different values?
   - Option: Take the value from the most recent (closest to output) call
   - Option: Require all values to match, error if they differ

2. **Missing lineage**: Records saved without lineage (raw data) can't be inferred
   - These would be skipped with a warning

3. **Schema migration**: After inference, the old parquet paths are still valid (they use schema keys only), but the metadata in SQLite needs updating

### API Sketch

```python
class DatabaseManager:
    def infer_versions(
        self,
        variable_class: Type[BaseVariable],
        rules: dict[str, dict | list[dict]],
        dry_run: bool = False,
        **schema_filter,
    ) -> dict:
        """
        Infer version keys from lineage for records with empty version.

        Args:
            variable_class: The type to update
            rules: Mapping of version_key -> extraction rules
            dry_run: If True, report what would change without modifying
            **schema_filter: Optional filter to limit which records to process

        Returns:
            Summary dict with counts of updated, skipped, failed records
        """
        ...
```

### Related: Manual Version Update

For simpler cases, a manual update method would also be useful:

```python
db.update_version(
    ProcessedSignal,
    where={"subject": 1, "visit": 2, "version": {}},
    set_version={"upstream": 2}
)
```

This directly sets version info without inferring from lineage.
