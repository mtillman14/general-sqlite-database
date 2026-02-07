# Quick Start

This guide will get you up and running with Thunk in a few minutes.

## Your First Thunk

Import the `thunk` decorator and apply it to any function:

```python
from thunk import thunk

@thunk
def double(x):
    return x * 2

result = double(5)
```

Notice that `result` is not `10` - it's an `ThunkOutput`:

```python
print(type(result))  # <class 'thunk.core.ThunkOutput'>
print(result.data)   # 10
```

## Accessing Lineage

Every `ThunkOutput` carries information about how it was computed:

```python
# The pipeline thunk captures the invocation
pt = result.pipeline_thunk

# See the captured inputs
print(pt.inputs)  # {'arg_0': 5}

# See the function that was called
print(pt.thunk.fcn.__name__)  # 'double'

# Get a unique hash for this computation
print(result.hash)  # SHA-256 based on function + inputs
```

## Chaining Computations

Thunks automatically track through chains of computations:

```python
@thunk
def add_one(x):
    return x + 1

@thunk
def multiply(x, y):
    return x * y

# Build a pipeline
a = add_one(5)       # 6
b = multiply(a, 3)   # 18

# b knows it came from multiply, which took output from add_one
print(b.pipeline_thunk.inputs['arg_0'])  # This is the ThunkOutput from add_one
```

## Multi-Output Functions

For functions returning multiple values:

```python
@thunk(unpack_output=True)
def split(data, pivot):
    return (
        [x for x in data if x < pivot],
        [x for x in data if x >= pivot]
    )

below, above = split([1, 5, 3, 8, 2], 4)
print(below.data)  # [1, 3, 2]
print(above.data)  # [5, 8]

# Each output has its own index
print(below.output_num)  # 0
print(above.output_num)  # 1
```

## Extracting Lineage Records

For storage or analysis, extract structured lineage:

```python
from thunk import extract_lineage

@thunk
def process(data, factor=2):
    return [x * factor for x in data]

result = process([1, 2, 3], factor=3)
lineage = extract_lineage(result)

print(lineage.function_name)  # 'process'
print(lineage.constants)      # [{'name': 'arg_0', ...}, {'name': 'factor', ...}]
```

## Next Steps

- [Basic Usage](../guide/basic-usage.md) - Deeper dive into features
- [Lineage Tracking](../guide/lineage.md) - Advanced lineage features
- [Caching](../guide/caching.md) - Configure caching backends
