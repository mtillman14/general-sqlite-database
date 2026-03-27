# SciLineage

**Lineage tracking for Python data pipelines.**

SciLineage is a lightweight library for building data processing pipelines with automatic provenance tracking. It captures the full computational lineage of your results, enabling reproducibility and intelligent caching.

## Features

- **Automatic Lineage Tracking**: Every computation captures its inputs and function, building a complete provenance graph
- **Input Classification**: Automatically distinguishes variable inputs from constants for accurate lineage
- **Pluggable Caching**: Register a backend via `configure_backend()` to enable cache lookups via lineage hashes
- **Lightweight**: Core dependency is only `canonicalhash`
- **Type Safe**: Full type hints throughout

## Installation

```bash
pip install scilineage
```

## Quick Start

### Basic Usage

```python
from scilineage import lineage_fcn

@lineage_fcn
def process(data, factor):
    return data * factor

# Call returns a LineageFcnResult, not the raw result
result = process([1, 2, 3], 2)

# Access the computed value
print(result.data)  # [2, 4, 6]

# Access lineage information
print(result.invoked.inputs)         # {'arg_0': [1, 2, 3], 'arg_1': 2}
print(result.invoked.fcn.fcn.__name__)  # 'process'
```

### Multi-Output Functions

```python
@lineage_fcn(unpack_output=True)
def split_data(data):
    mid = len(data) // 2
    return data[:mid], data[mid:]

first, second = split_data([1, 2, 3, 4])
print(first.data)   # [1, 2]
print(second.data)  # [3, 4]
```

### Chaining Computations

```python
@lineage_fcn
def normalize(data):
    max_val = max(data)
    return [x / max_val for x in data]

@lineage_fcn
def scale(data, factor):
    return [x * factor for x in data]

raw = [10, 20, 30, 40]
normalized = normalize(raw)
scaled = scale(normalized, 100)

print(scaled.data)  # [25.0, 50.0, 75.0, 100.0]
```

### Extracting Lineage

```python
from scilineage import extract_lineage, get_upstream_lineage

@lineage_fcn
def step1(x):
    return x + 1

@lineage_fcn
def step2(x):
    return x * 2

result = step2(step1(5))

lineage = extract_lineage(result)
print(lineage.function_name)   # 'step2'
print(lineage.function_hash)   # SHA-256 of function bytecode

chain = get_upstream_lineage(result)
for record in chain:
    print(f"{record['function_name']}: inputs={record['inputs']}")
```

### Manual Interventions

```python
from scilineage import manual

# Step outside the pipeline for a manual correction
edited_data = [1, 2, 3]

# Re-enter the pipeline — the intervention is documented in lineage
corrected = manual(
    edited_data,
    label="outlier_removal",
    reason="amplitude < 0.1 in trial 3 is sensor artifact",
)
```

## API Reference

### `@lineage_fcn(unpack_output=False, unwrap=True, generates_file=False)`

Decorator to convert a function into a `LineageFcn`.

- `unpack_output`: Whether to unpack a tuple return into separate `LineageFcnResult`s
- `unwrap`: If True, automatically unwrap `LineageFcnResult` inputs to their raw data
- `generates_file`: If True, marks the function as producing files as side effects

### `LineageFcnResult`

Wrapper around computed values that carries lineage.

- `.data`: The actual computed value
- `.invoked`: The `LineageFcnInvocation` that produced this
- `.hash`: Unique hash based on computation lineage
- `.output_num`: Index for multi-output functions

### `LineageFcnInvocation`

Represents a specific function invocation with captured inputs.

- `.fcn`: The parent `LineageFcn` (function wrapper)
- `.inputs`: Dict of captured input values
- `.outputs`: Tuple of `LineageFcnResult` results
- `.compute_lineage_hash()`: Generate lineage hash for cache key computation

### `LineageFcn`

The decorated function wrapper.

- `.fcn`: The original wrapped function
- `.hash`: SHA-256 hash of function bytecode
- `.invocations`: All `LineageFcnInvocation`s created from this

## License

MIT License - see [LICENSE](LICENSE) for details.
