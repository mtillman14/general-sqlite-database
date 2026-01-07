# Computation Caching

SciDB automatically caches computation results. When you save a variable produced by a thunked function, the result is cached. Future identical computations are skipped automatically.

## How Caching Works

1. **Save populates cache** - When saving an `OutputThunk`, the cache is updated
2. **Cache key** - Hash of function + input hashes
3. **Automatic lookup** - Thunks check the cache before executing

```
                    ┌─────────────────┐
                    │ @thunk function │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────┐          ┌─────────────────┐
    │  First run:     │          │  Later run:     │
    │  Execute +      │          │  Auto cache hit │
    │  Save + Cache   │          │  Skip execution │
    └─────────────────┘          └─────────────────┘
```

## Automatic Caching

Caching is fully automatic. Just save once, and future calls with the same inputs skip execution:

```python
@thunk(n_outputs=1)
def expensive_computation(data):
    print("Computing...")  # Only prints on first run
    return data * 2

# First run: executes and prints "Computing..."
result = expensive_computation(raw_data)
MyVar(result).save(subject=1, stage="computed")

# Second run: cache hit, no execution!
result2 = expensive_computation(raw_data)  # No print, returns cached
print(result2.was_cached)  # True
print(result2.data)        # Same result, no recomputation
```

**Requirements for automatic caching:**
- Database must be configured (`configure_database(...)`)
- Variable class must be registered (happens on first `save()` or `load()`)

### Multi-Output Functions

Multi-output functions are also cached automatically. All outputs must be saved before caching takes effect:

```python
@thunk(n_outputs=2)
def split_data(data):
    print("Splitting...")  # Only prints on first run
    return data[:len(data)//2], data[len(data)//2:]

# First run: executes
left, right = split_data(raw_data)
LeftHalf(left).save(subject=1)
RightHalf(right).save(subject=1)

# Second run: cache hit for both outputs!
left2, right2 = split_data(raw_data)  # No print
print(left2.was_cached)   # True
print(right2.was_cached)  # True
```

**Important:** If only some outputs are saved, no caching occurs:

```python
left, right = split_data(raw_data)
LeftHalf(left).save(subject=1)  # Only save one output
# right is not saved

# Next run: cache miss (partial save)
left2, right2 = split_data(raw_data)  # Executes again
print(left2.was_cached)  # False
```

## Manual Cache Checking

For more control, use `check_cache()` explicitly:

```python
from scidb import check_cache

@thunk(n_outputs=1)
def process(data):
    return data * 2

result = process(input_data)
cached = check_cache(result.pipeline_thunk, OutputVar, db=db)

if cached:
    print(f"Cache hit! vhash: {cached.cached_id}")
else:
    print("Cache miss, saving...")
    OutputVar(result).save(db=db, subject=1)
```

## Cache Key Components

The cache key is a SHA-256 hash of:

| Component | Source |
|-----------|--------|
| Function hash | Bytecode + constants |
| Input vhashes | For saved variables |
| Input content hashes | For unsaved values |
| Output thunk hashes | For chained computations |

This ensures cache hits only when:

- Same function code
- Same input data
- Same input metadata

## Cache Statistics

```python
stats = db.get_cache_stats()

print(f"Total cached: {stats['total_entries']}")
print(f"Functions: {stats['functions']}")
for fn, count in stats['entries_by_function'].items():
    print(f"  {fn}: {count} entries")
```

## Cache Invalidation

Cache invalidation is primarily useful for:
- **Space management**: Removing old entries you no longer need
- **Testing**: Forcing recomputation during development
- **Cleanup**: Removing entries from functions that no longer exist

Note: If you fix a bug in a function, the function's bytecode hash changes automatically, so old cached results won't be hit. You don't need to manually invalidate.

### Invalidate All

```python
deleted = db.invalidate_cache()
print(f"Removed {deleted} cache entries")
```

### Invalidate by Function Name

```python
# Remove all cached results from a specific function
deleted = db.invalidate_cache(function_name="process_signal")
```

### Invalidate by Function Hash

Remove entries for a specific version of a function:

```python
deleted = db.invalidate_cache(function_hash="abc123...")
```

## OutputThunk Cache Properties

```python
result = expensive_computation(data)
cached = check_cache(result.pipeline_thunk, MyVar, db=db)

if cached:
    cached.was_cached     # True
    cached.cached_id   # "abc123..."
    cached.data           # The cached data
```

## When Cache Misses Occur

Cache misses happen when:

| Scenario | Reason |
|----------|--------|
| First run | No previous result exists |
| Different inputs | Input data changed |
| Function modified | Bytecode hash changed |
| Different constants | e.g., `x * 2` vs `x * 3` |
| Cache invalidated | Manual invalidation |

## Best Practices

### 1. Save After Expensive Computations

```python
result = expensive_computation(data)
MyVar(result).save(db=db, subject=1)  # Populates cache
```

### 2. Check Cache Before Re-running

```python
cached = check_cache(pipeline_thunk, MyVar, db=db)
if cached:
    return cached.data
```

### 3. Use Saved Variables as Inputs

Variables with vhashes have stable cache keys:

```python
# Good: loaded variable has vhash
raw = RawData.load(subject=1)
result = process(raw.data)

# Less stable: unsaved data uses content hash
result = process(np.array([1, 2, 3]))
```

### 4. Cache Keys Are Content-Based

If you modify a function's code, the cache key changes automatically. You don't need to manually invalidate—the next run will simply compute fresh results.
