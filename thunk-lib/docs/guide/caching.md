# Caching

Thunk supports pluggable caching backends to avoid redundant computations. When a cache is configured, thunked functions automatically check for cached results before executing.

## Configuring a Cache Backend

Implement the `CacheBackend` protocol and register it:

```python
from thunk import configure_cache

class MyCache:
    def __init__(self):
        self.store = {}

    def get_cached(self, cache_key: str, n_outputs: int):
        """
        Look up cached results.

        Args:
            cache_key: SHA-256 hash identifying the computation
            n_outputs: Number of outputs expected

        Returns:
            List of (data, identifier) tuples if ALL outputs cached,
            None otherwise.
        """
        if cache_key in self.store:
            results = self.store[cache_key]
            if len(results) == n_outputs:
                return results
        return None

    def save(self, cache_key: str, results: list):
        """Save results to cache."""
        self.store[cache_key] = results

cache = MyCache()
configure_cache(cache)
```

## Cache Key Computation

Cache keys are computed from:

1. **Function hash**: SHA-256 of bytecode + constants
2. **Input lineage**: For each input:
   - `ThunkOutput`: Uses its lineage-based hash
   - Trackable variable: Uses lineage_hash if available
   - Raw value: Uses content hash

This means:

- Same function + same inputs = same cache key
- Same content but different computation = different cache key

```python
@thunk
def process(x):
    return x * 2

result1 = process(5)
result2 = process(5)

# Same cache key
key1 = result1.pipeline_thunk.compute_cache_key()
key2 = result2.pipeline_thunk.compute_cache_key()
assert key1 == key2
```

## Cache Hit Behavior

When a cache hit occurs:

```python
from thunk import thunk, configure_cache

class SimpleCache:
    def __init__(self):
        self.store = {}
        self.hits = 0

    def get_cached(self, cache_key, n_outputs):
        if cache_key in self.store:
            self.hits += 1
            return self.store[cache_key]
        return None

    def save(self, cache_key, result, identifier="cached"):
        self.store[cache_key] = [(result, identifier)]

cache = SimpleCache()
configure_cache(cache)

@thunk
def expensive(x):
    print("Computing...")
    return x ** 2

# First call - computes
r1 = expensive(5)
cache.save(r1.pipeline_thunk.compute_cache_key(), r1.data)

# Second call - cache hit (no "Computing..." printed)
r2 = expensive(5)
print(r2.was_cached)  # True
print(cache.hits)     # 1
```

## Checking Cache Status

```python
result = expensive(5)

if result.was_cached:
    print(f"Loaded from cache: {result.cached_id}")
else:
    print("Freshly computed")
```

## Disabling Cache

```python
from thunk import configure_cache

# Disable caching
configure_cache(None)
```

## Integration with SciDB

SciDB automatically registers itself as a cache backend:

```python
from scidb import configure_database
from thunk import thunk

# This automatically configures thunk's cache
db = configure_database("experiment.db")

@thunk
def process(data):
    return data * 2

# Results are automatically cached in the database
result = process(input_data)
```

## Custom Cache Backend Example

### Redis Cache

```python
import json
import redis
from thunk import configure_cache

class RedisCache:
    def __init__(self, host='localhost', port=6379):
        self.client = redis.Redis(host=host, port=port)

    def get_cached(self, cache_key, n_outputs):
        data = self.client.get(f"thunk:{cache_key}")
        if data:
            results = json.loads(data)
            if len(results) == n_outputs:
                return [(r['data'], r['id']) for r in results]
        return None

    def save(self, cache_key, data, identifier):
        # Would need serialization strategy for complex data
        pass

configure_cache(RedisCache())
```

### File-Based Cache

```python
import hashlib
import pickle
from pathlib import Path
from thunk import configure_cache

class FileCache:
    def __init__(self, cache_dir=".thunk_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _path(self, cache_key):
        return self.cache_dir / f"{cache_key}.pkl"

    def get_cached(self, cache_key, n_outputs):
        path = self._path(cache_key)
        if path.exists():
            with open(path, 'rb') as f:
                results = pickle.load(f)
            if len(results) == n_outputs:
                return results
        return None

    def save(self, cache_key, results):
        with open(self._path(cache_key), 'wb') as f:
            pickle.dump(results, f)

configure_cache(FileCache())
```

## Best Practices

1. **Cache expensive computations**: Focus on functions that take significant time
2. **Consider cache invalidation**: Function changes should invalidate old cache entries
3. **Handle cache misses gracefully**: Always have a fallback to recompute
4. **Monitor cache hit rates**: Track effectiveness of your caching strategy
