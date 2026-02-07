# Thunk

**Lazy evaluation and lineage tracking for Python data pipelines.**

Thunk is a lightweight library inspired by Haskell's thunk concept, designed for building data processing pipelines with automatic provenance tracking.

## Why Thunk?

When building data pipelines, you often need to:

1. **Track provenance**: Know exactly how a result was computed
2. **Enable reproducibility**: Re-run computations with the same inputs
3. **Avoid redundant work**: Cache expensive computations
4. **Debug pipelines**: Inspect intermediate values and their origins

Thunk solves all of these by wrapping your functions to automatically capture their inputs and outputs, building a complete lineage graph.

## Quick Example

```python
from thunk import thunk

@thunk
def process(data, factor):
    return data * factor

result = process([1, 2, 3], 2)

# Access computed value
print(result.data)  # [2, 4, 6]

# Access lineage
print(result.pipeline_thunk.inputs)
# {'arg_0': [1, 2, 3], 'arg_1': 2}
```

## Features

- **Zero Configuration**: Just add `@thunk` to your functions
- **Automatic Lineage**: Full provenance graph captured automatically
- **Pluggable Caching**: Bring your own cache backend
- **Lightweight**: No heavy dependencies required
- **Type Safe**: Full type hints and PEP 561 compliance

## Installation

```bash
pip install thunk
```

## Next Steps

- [Installation](getting-started/installation.md) - Detailed installation instructions
- [Quick Start](getting-started/quickstart.md) - Get up and running in minutes
- [Basic Usage](guide/basic-usage.md) - Learn the fundamentals
- [API Reference](api/core.md) - Complete API documentation
