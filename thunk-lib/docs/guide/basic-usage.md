# Basic Usage

## The `@thunk` Decorator

The `@thunk` decorator transforms a regular function into a `Thunk` object:

```python
from thunk import thunk

@thunk(n_outputs=1)
def process(data):
    return data * 2
```

### Parameters

- **`n_outputs`** (default: 1): Number of values the function returns
- **`unwrap`** (default: True): Whether to unwrap `OutputThunk` inputs automatically

### Understanding `unwrap`

By default, if you pass an `OutputThunk` to a thunked function, it automatically extracts the `.data`:

```python
@thunk(n_outputs=1)
def step1(x):
    return x + 1

@thunk(n_outputs=1)
def step2(x):
    # x is automatically the raw value (6), not the OutputThunk
    return x * 2

result = step2(step1(5))  # (5+1) * 2 = 12
```

With `unwrap=False`, you receive the full `OutputThunk`:

```python
@thunk(n_outputs=1, unwrap=False)
def debug_step(x):
    print(f"Received: {type(x)}")  # OutputThunk
    print(f"Value: {x.data}")
    print(f"Hash: {x.hash}")
    return x.data * 2
```

## Working with OutputThunk

Every thunked function call returns an `OutputThunk`:

```python
result = process([1, 2, 3])

# Access the computed value
value = result.data

# Check if computation completed
print(result.is_complete)  # True

# Get unique identifier
print(result.hash)

# Check cache status
print(result.was_cached)  # False (first computation)
```

## PipelineThunk: The Computation Record

Each `OutputThunk` links to a `PipelineThunk` that records the invocation:

```python
pt = result.pipeline_thunk

# Captured inputs (both args and kwargs)
print(pt.inputs)
# {'arg_0': [1, 2, 3]}

# Parent Thunk (the function wrapper)
print(pt.thunk.fcn.__name__)  # 'process'

# Generate cache key
cache_key = pt.compute_cache_key()
```

## Multi-Output Functions

When a function returns multiple values:

```python
@thunk(n_outputs=3)
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
@thunk(n_outputs=1)
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
@thunk(n_outputs=1)
def risky(x):
    if x < 0:
        raise ValueError("Negative input")
    return x ** 0.5

try:
    result = risky(-1)
except ValueError as e:
    print(e)  # "Negative input"
```

If `n_outputs` doesn't match the actual return:

```python
@thunk(n_outputs=2)
def wrong():
    return 42  # Returns 1 value, expected 2

result = wrong()  # Raises ValueError
```
