# Thunk

**Lazy evaluation and lineage tracking for Python data pipelines.**

Thunk is a lightweight library inspired by Haskell's thunk concept, designed for building data processing pipelines with automatic provenance tracking. It captures the full computational lineage of your results, enabling reproducibility and intelligent caching.

## Features

- **Automatic Lineage Tracking**: Every computation captures its inputs and function, building a complete provenance graph
- **Lazy Evaluation**: Results are wrapped in `ThunkOutput` objects that carry lineage information
- **Pluggable Caching**: Configure custom cache backends to avoid redundant computations
- **Zero Heavy Dependencies**: Core functionality works without numpy/pandas (optional integrations available)
- **Type Safe**: Full type hints and PEP 561 compliance

## Installation

```bash
pip install thunk
```

With optional dependencies:

```bash
pip install thunk[numpy]     # numpy support
pip install thunk[pandas]    # pandas support
pip install thunk[all]       # all optional dependencies
```

## Quick Start

### Basic Usage

```python
from thunk import thunk

@thunk(n_outputs=1)
def process(data, factor):
    return data * factor

# Call returns an ThunkOutput, not the raw result
result = process([1, 2, 3], 2)

# Access the computed value
print(result.data)  # [2, 4, 6]

# Access lineage information
print(result.pipeline_thunk.inputs)  # {'arg_0': [1, 2, 3], 'arg_1': 2}
print(result.pipeline_thunk.thunk.fcn.__name__)  # 'process'
```

### Multi-Output Functions

```python
@thunk(n_outputs=2)
def split_data(data):
    mid = len(data) // 2
    return data[:mid], data[mid:]

first, second = split_data([1, 2, 3, 4])
print(first.data)   # [1, 2]
print(second.data)  # [3, 4]
```

### Chaining Computations

```python
@thunk(n_outputs=1)
def normalize(data):
    max_val = max(data)
    return [x / max_val for x in data]

@thunk(n_outputs=1)
def scale(data, factor):
    return [x * factor for x in data]

# Build a pipeline - lineage is automatically tracked
raw = [10, 20, 30, 40]
normalized = normalize(raw)
scaled = scale(normalized, 100)

# The full computation graph is captured
print(scaled.data)  # [25.0, 50.0, 75.0, 100.0]
```

### Extracting Lineage

```python
from thunk import extract_lineage, get_lineage_chain

@thunk(n_outputs=1)
def step1(x):
    return x + 1

@thunk(n_outputs=1)
def step2(x):
    return x * 2

result = step2(step1(5))

# Get immediate lineage
lineage = extract_lineage(result)
print(lineage.function_name)  # 'step2'
print(lineage.function_hash)  # SHA-256 of function bytecode

# Get full lineage chain
chain = get_lineage_chain(result)
for record in chain:
    print(f"{record.function_name}: {record.inputs}, {record.constants}")
```

### Configuring Caching

```python
from thunk import configure_cache, CacheBackend

class MyCache:
    def __init__(self):
        self.store = {}

    def get_cached(self, cache_key: str, n_outputs: int):
        """Return list of (data, id) tuples or None."""
        if cache_key in self.store:
            return self.store[cache_key]
        return None

    def save(self, cache_key: str, results):
        self.store[cache_key] = results

cache = MyCache()
configure_cache(cache)

# Now repeated calls with same inputs will check the cache
```

## API Reference

### `@thunk(n_outputs=1, unwrap=True)`

Decorator to convert a function into a Thunk.

- `n_outputs`: Number of return values (default: 1)
- `unwrap`: If True, automatically unwrap `ThunkOutput` inputs to their raw data

### `ThunkOutput`

Wrapper around computed values that carries lineage.

- `.data`: The actual computed value
- `.pipeline_thunk`: The `PipelineThunk` that produced this
- `.hash`: Unique hash based on computation lineage
- `.was_cached`: True if loaded from cache
- `.output_num`: Index for multi-output functions

### `PipelineThunk`

Represents a specific function invocation with captured inputs.

- `.thunk`: The parent `Thunk` (function wrapper)
- `.inputs`: Dict of captured input values
- `.outputs`: Tuple of `ThunkOutput` results
- `.compute_cache_key()`: Generate cache key based on lineage

### `LineageRecord`

Structured provenance information.

- `.function_name`: Name of the function
- `.function_hash`: Hash of function bytecode
- `.inputs`: List of input descriptors (variables)
- `.constants`: List of constant values

### Utility Functions

- `extract_lineage(thunk_output)`: Get `LineageRecord` for an output
- `get_lineage_chain(thunk_output)`: Get full lineage history
- `get_raw_value(data)`: Unwrap `ThunkOutput` or return as-is
- `canonical_hash(obj)`: Deterministic hash for any Python object
- `configure_cache(backend)`: Set global cache backend

## Integration with SciDB

Thunk is designed to work seamlessly with [SciDB](https://github.com/example/scidb), a scientific data versioning framework. When used together:

```python
from scidb import configure_database, BaseVariable
from thunk import thunk

db = configure_database("experiment.db")

class MyData(BaseVariable):
    # ... implementation

@thunk(n_outputs=1)
def process(data):
    return data * 2

# SciDB automatically registers as a cache backend
result = process(loaded_data)
MyData(result).save(db=db, experiment=1)  # Lineage is captured
```

## License

MIT License - see [LICENSE](LICENSE) for details.
