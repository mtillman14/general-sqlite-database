# Basic Usage

## The `@thunk` Decorator

The `@thunk` decorator transforms a regular function into a `Thunk` object:

```python
from thunk import thunk

@thunk
def process(data):
    return data * 2
```

### Parameters

- **`unpack_output`** (default: False): Whether to unpack a returned tuple into separate ThunkOutputs.
- **`unwrap`** (default: True): Whether to unwrap `ThunkOutput` inputs automatically

### Understanding `unwrap`

By default, if you pass an `ThunkOutput` to a thunked function, it automatically extracts the `.data`:

```python
@thunk
def step1(x):
    return x + 1

@thunk
def step2(x):
    # x is automatically the raw value (6), not the ThunkOutput
    return x * 2

result = step2(step1(5))  # (5+1) * 2 = 12
```

With `unwrap=False`, you receive the full `ThunkOutput`:

```python
@thunk(unwrap=False)
def debug_step(x):
    print(f"Received: {type(x)}")  # ThunkOutput
    print(f"Value: {x.data}")
    print(f"Hash: {x.hash}")
    return x.data * 2
```

## Working with ThunkOutput

Every thunked function call returns an `ThunkOutput`:

```python
result = process([1, 2, 3])

# Access the computed value
value = result.data

# Check if computation completed
print(result.is_complete)  # True

# Get unique identifier
print(result.hash)

# Get output index (for multi-output functions)
print(result.output_num)  # 0
```

## PipelineThunk: The Computation Record

Each `ThunkOutput` links to a `PipelineThunk` that records the invocation:

```python
pt = result.pipeline_thunk

# Captured inputs (both args and kwargs)
print(pt.inputs)
# {'arg_0': [1, 2, 3]}

# Parent Thunk (the function wrapper)
print(pt.thunk.fcn.__name__)  # 'process'

# Generate lineage hash (used for cache key computation)
lineage_hash = pt.compute_lineage_hash()
```

## Multi-Output Functions

When a function returns multiple values:

```python
@thunk(unpack_output=True)
def analyze(data):
    return min(data), max(data), sum(data) / len(data)

minimum, maximum, average = analyze([1, 2, 3, 4, 5])

print(minimum.data)  # 1
print(maximum.data)  # 5
print(average.data)  # 3.0

# All share the same pipeline thunk
assert minimum.pipeline_thunk is maximum.pipeline_thunk

# But have different output indices
print(minimum.output_num)  # 0
print(maximum.output_num)  # 1
print(average.output_num)  # 2
```

## Hash-Based Identity

Thunks use content-based hashing for identity:

```python
@thunk
def compute(x):
    return x * 2

# Same inputs = same hash
r1 = compute(5)
r2 = compute(5)
print(r1.hash == r2.hash)  # True

# Different inputs = different hash
r3 = compute(6)
print(r1.hash == r3.hash)  # False
```

This enables intelligent caching based on computation identity, not just input values.

## Error Handling

If a thunked function raises an exception, it propagates normally:

```python
@thunk
def risky(x):
    if x < 0:
        raise ValueError("Negative input")
    return x ** 0.5

try:
    result = risky(-1)
except ValueError as e:
    print(e)  # "Negative input"
```
