# Schema ID Allocation

## Overview

Each unique combination of schema key values gets a row in the `_schema` table with an integer `schema_id` (primary key). This ID is used as a foreign key in variable data tables. There are two code paths that allocate these IDs, and they must stay consistent to avoid primary key collisions.

## The `_schema` Table

```sql
CREATE TABLE _schema (
    schema_id INTEGER PRIMARY KEY,
    schema_level VARCHAR NOT NULL,
    -- one VARCHAR column per schema key (e.g. subject, session, trial)
)
```

## ID Allocation Strategy: `MAX(schema_id) + 1`

Both paths use the same strategy — `SELECT COALESCE(MAX(schema_id), 0) + 1 FROM _schema` — to allocate the next ID. This is safe for a single-user desktop application (no concurrent writes).

### Single-record path: `_get_or_create_schema_id()` (sciduck.py)

Called by `DatabaseManager.save()` → `_save_native_with_schema()` and `_save_custom_to_sciduck()`. Flow:

1. SELECT existing row matching schema_level + key values + NULLs for missing keys
2. If found → return existing `schema_id` (no insert)
3. If not found → `MAX(schema_id) + 1` → INSERT → return new ID

### Batch path: `batch_get_or_create_schema_ids()` (sciduck.py)

Called by `DatabaseManager.save_batch()`. Flow:

1. SELECT all existing rows for the given schema_level
2. Match in Python against requested combos
3. For missing combos: allocate a contiguous block starting at `MAX(schema_id) + 1`
4. Batch INSERT via DataFrame

## Historical Bug: Sequence Desync (Fixed)

Previously, the single-record path used a DuckDB sequence (`nextval('_schema_id_seq')`) while the batch path used `MAX+1` and tried to resync the sequence afterward. The resync was wrapped in `try/except: pass`, so it could fail silently. When it did, subsequent single-record saves would get already-used IDs from the stale sequence, causing `Duplicate key "schema_id: N" violates primary key constraint`.

The fix was to make both paths use `MAX+1`, eliminating the sequence entirely. The `_schema_id_seq` sequence still exists in the DDL (for backward compatibility with existing databases) but is no longer read by either path.

## Idempotency

Schema ID allocation is inherently idempotent — `_get_or_create_schema_id` returns the existing ID if a matching row exists. The broader save idempotency comes from `DatabaseManager.save()` which checks `_record_metadata` for an existing `record_id` (a deterministic hash of class name, schema version, content hash, and metadata) and returns early if found. This means re-running the same save with identical data skips all the way past schema resolution.

## Test Coverage

Regression tests in `sciduck/tests/test_sciduck.py`:
- `test_batch_then_single_schema_id_no_collision` — batch creates entries, then single-record create must not collide
- `test_single_then_batch_schema_id_no_collision` — single-record create, then batch must reuse the existing ID and not collide on new ones
