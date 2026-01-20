# API Changes Summary

## Current Task: Refactor `BaseVariable.save()` from instance method to classmethod

### The Change

**Before (instance method):**
```python
var = MyVariable(data)
var.save(db=db, subject=1, trial=1)

# Or combined:
MyVariable(data).save(subject=1, trial=1)
```

**After (classmethod):**
```python
MyVariable.save(data, db=db, subject=1, trial=1)
```

### Why This Change

1. Cleaner API - no need to create instance first
2. Uniform interface - `save` and `load` are both classmethods
3. Handles multiple input types transparently:
   - `OutputThunk` (from @thunk computations) - extracts data + lineage
   - `BaseVariable` instance - extracts data, preserves lineage
   - Raw data (numpy array, etc.) - wraps directly

### Implementation Status

**Completed:**
- [x] Changed `BaseVariable.save()` to classmethod in `src/scidb/variable.py`
- [x] Updated `save_from_dataframe()` to use new API
- [x] Parquet file storage integration (separate change, also in progress)

**In Progress:**
- [ ] Update source code docstrings/examples
- [ ] Update all test files
- [ ] Update documentation
- [ ] Update example files

### Files Requiring Updates

#### Source Code (docstrings/examples)
- `src/scidb/__init__.py` - lines 32, 44
- `src/scidb/database.py` - line 110 (docstring example)
- `src/scidb/lineage.py` - line 61 (docstring example)

#### Tests (~100+ locations)
- `tests/test_variable.py`
- `tests/test_database.py`
- `tests/test_integration.py`
- `tests/test_lineage.py`
- `tests/test_lineage_mode.py`
- `tests/test_caching.py`
- `tests/test_thunk.py`
- `tests/test_preview.py`
- `tests/test_query.py`

#### Examples
- `examples/example1/example1_main.py`
- `examples/pipeline_demo.py`
- `examples/pipeline_example1.py`
- `examples/debug_example2_advanced.py`
- `spec/code_examples/example1.py`

### Transformation Patterns

1. **Simple instance save:**
   ```python
   # Before
   var = MyClass(data)
   var.save(subject=1)

   # After
   MyClass.save(data, subject=1)
   ```

2. **Inline construction and save:**
   ```python
   # Before
   MyClass(data).save(db=db, subject=1)

   # After
   MyClass.save(data, db=db, subject=1)
   ```

3. **Save with explicit db:**
   ```python
   # Before
   var.save(db=db, subject=1, trial=1)

   # After
   MyClass.save(var.data, db=db, subject=1, trial=1)
   # Or if var is a BaseVariable instance:
   MyClass.save(var, db=db, subject=1, trial=1)
   ```

4. **Save OutputThunk from @thunk:**
   ```python
   # Before
   result = process(input)  # OutputThunk
   MyClass(result).save(subject=1)

   # After
   result = process(input)  # OutputThunk
   MyClass.save(result, subject=1)
   ```

5. **Re-save loaded variable with new metadata:**
   ```python
   # Before
   var = MyClass.load(subject=1, trial=1)
   var.save(subject=1, trial=2)  # Different metadata

   # After
   var = MyClass.load(subject=1, trial=1)
   MyClass.save(var, subject=1, trial=2)
   ```

### Related Changes (Already Implemented)

#### Parquet File Storage
- Data now stored as Parquet files instead of SQLite BLOBs
- Path structure: `parquet/<table_name>/<metadata_hierarchy>/<vhash>.parquet`
- SQLite stores metadata only, references Parquet files
- DuckDB query interface available via `src/scidb/query.py`

### Notes

- Pre-alpha status, no backwards compatibility needed
- `load()` is already a classmethod, so this makes the API symmetric
- The new `save()` returns vhash (same as before)
