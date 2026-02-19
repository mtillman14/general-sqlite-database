# DuckDB Transaction Safety in save()

## Context

The `DatabaseManager.save()` method in `src/scidb/database.py` wraps all writes
(data INSERT, `_record_metadata` INSERT, lineage save) in an explicit DuckDB
transaction (`_begin()` / `_commit()`) to avoid repeated WAL checkpoints that
cause multi-second stalls.

## The Problem

DuckDB enforces strict transaction state: once any statement within a `BEGIN
TRANSACTION` fails, the transaction enters an **aborted** state. Every
subsequent statement on that connection — including simple SELECTs — will fail
with:

    TransactionContext Error: Current transaction is aborted (please ROLLBACK)

This persists until an explicit `ROLLBACK` is issued.

## Why This Matters for `for_each` with `distribute=`

The `distribute=True` mode in `for_each` saves multiple rows in a loop (one per
distributed element). Each row calls `save()` independently. Without rollback
handling:

1. Row N's save fails mid-transaction (e.g., type mismatch, schema conflict)
2. The exception propagates to `for_each`'s try/catch, which logs `[error]`
3. Row N+1's save attempt fails at the **idempotency check SELECT** (before
   `_begin()` is even called) because the connection is poisoned
4. All remaining rows fail with the same "please ROLLBACK" message
5. The real error (from row N) is buried under dozens of identical cascade errors

## The Fix

`save()` wraps the transaction body in `try/except`. On failure, it calls
`_rollback()` before re-raising:

```python
self._duck._begin()
try:
    # ... all write operations ...
    self._duck._commit()
except Exception:
    try:
        self._duck._rollback()
    except Exception:
        pass  # Connection may already be closed
    raise
```

The `_rollback()` method was added to `SciDuck` in `sciduck/src/sciduck/sciduck.py`.

## Key Takeaway

Any code path that calls `_begin()` must guarantee either `_commit()` or
`_rollback()` executes — never leave an explicit transaction open. This is
especially critical when the caller (like `for_each`) catches errors and
continues using the same database connection.
