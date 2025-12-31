# Computation Caching

SciDB automatically caches computation results. When you save a variable produced by a thunked function, the result is cached. Future identical computations can be skipped.

## How Caching Works

1. **Save populates cache** - When saving an `OutputThunk`, the cache is updated
2. **Cache key** - Hash of function + input hashes
3. **Cache lookup** - Check before re-running expensive computations

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
    │  Execute +      │          │  Check cache    │
    │  Save + Cache   │          │  Return cached  │
    └─────────────────┘          └─────────────────┘
```

## Automatic Cache Population

Cache is populated automatically when saving thunk results:

```python
@thunk(n_outputs=1)
def expensive_computation(data):
    # ... takes 10 minutes
    return result

# First run: executes and caches
result = expensive_computation(raw_data)
MyVar(result).save(subject=1, stage="computed")

# Later: same inputs -> cache hit
result2 = expensive_computation(raw_data)  # Still executes
cached = check_cache(result2.pipeline_thunk, MyVar, db=db)
if cached:
    # Use cached result instead
    print(f"Cache hit! vhash: {cached.cached_vhash}")
```

## Explicit Cache Checking

Use `check_cache()` before running expensive computations:

```python
from scidb import check_cache, PipelineThunk

@thunk(n_outputs=1)
def process(data):
    return data * 2

# Create the pipeline thunk
result = process(input_data)

# Check if already cached
cached = check_cache(result.pipeline_thunk, OutputVar, db=db)
if cached:
    print("Using cached result")
    final = cached.data
else:
    print("Computing...")
    final = result.data
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
    cached.cached_vhash   # "abc123..."
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
