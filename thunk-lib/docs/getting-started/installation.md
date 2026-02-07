# Installation

## Requirements

- Python 3.10 or higher

## Basic Installation

Install thunk using pip:

```bash
pip install thunk
```

## Optional Dependencies

Thunk's core dependency is `canonicalhash`. It also provides enhanced support for scientific computing libraries as optional extras:

### NumPy Support

For optimal hashing of numpy arrays:

```bash
pip install thunk[numpy]
```

### Pandas Support

For optimal hashing of DataFrames and Series:

```bash
pip install thunk[pandas]
```

### All Optional Dependencies

```bash
pip install thunk[all]
```

## Development Installation

For contributing to thunk:

```bash
git clone https://github.com/example/thunk.git
cd thunk
pip install -e ".[dev]"
```

## Verifying Installation

```python
from thunk import thunk, __version__

print(f"Thunk version: {__version__}")

@thunk
def test(x):
    return x * 2

result = test(5)
print(f"Result: {result.data}")  # Should print: Result: 10
```
