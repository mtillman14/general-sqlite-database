# Lineage Tracking

Thunk automatically tracks the computational lineage of every result, enabling full reproducibility and provenance analysis.

## Understanding Lineage

When you chain thunked functions, each `OutputThunk` maintains a reference to:

1. The `PipelineThunk` that created it (which function, which inputs)
2. Any `OutputThunk` inputs (creating a graph of computations)

```python
from thunk import thunk

@thunk(n_outputs=1)
def load(path):
    return [1, 2, 3]  # Simulated data loading

@thunk(n_outputs=1)
def normalize(data):
    max_val = max(data)
    return [x / max_val for x in data]

@thunk(n_outputs=1)
def scale(data, factor):
    return [x * factor for x in data]

# Build computation graph
raw = load("data.csv")
normed = normalize(raw)
scaled = scale(normed, 100)

# scaled knows its full history
```

## Extracting Lineage Records

Use `extract_lineage()` to get a structured `LineageRecord`:

```python
from thunk import extract_lineage

lineage = extract_lineage(scaled)

print(lineage.function_name)   # 'scale'
print(lineage.function_hash)   # SHA-256 of function bytecode

# Inputs that came from other thunks
for inp in lineage.inputs:
    print(inp)
# {'name': 'arg_0', 'source_type': 'thunk', 'source_function': 'normalize', ...}

# Literal values (constants)
for const in lineage.constants:
    print(const)
# {'name': 'arg_1', 'value_hash': '...', 'value_repr': '100', 'value_type': 'int'}
```

## Full Lineage Chain

To traverse the entire computation history:

```python
from thunk import get_lineage_chain

chain = get_lineage_chain(scaled)

for record in chain:
    print(f"{record.function_name}:")
    print(f"  Inputs: {len(record.inputs)}")
    print(f"  Constants: {len(record.constants)}")

# Output:
# scale:
#   Inputs: 1
#   Constants: 1
# normalize:
#   Inputs: 1
#   Constants: 0
# load:
#   Inputs: 0
#   Constants: 1
```

## LineageRecord Structure

```python
@dataclass
class LineageRecord:
    function_name: str      # Name of the function
    function_hash: str      # SHA-256 of bytecode
    inputs: list[dict]      # Variable inputs (from thunks or saved variables)
    constants: list[dict]   # Literal values
```

### Input Descriptors

Inputs from other thunks:
```python
{
    "name": "arg_0",
    "source_type": "thunk",
    "source_function": "normalize",
    "source_hash": "abc123...",
    "output_num": 0
}
```

Inputs from trackable variables (e.g., scidb):
```python
{
    "name": "data",
    "source_type": "variable",
    "type": "MyDataClass",
    "record_id": "def456...",
    "metadata": {"subject": 1}
}
```

### Constant Descriptors

```python
{
    "name": "factor",
    "value_hash": "789abc...",
    "value_repr": "100",
    "value_type": "int"
}
```

## Serialization

LineageRecords can be serialized for storage:

```python
# To dictionary
data = lineage.to_dict()

# From dictionary
restored = LineageRecord.from_dict(data)
```

## Use Cases

### Reproducibility Verification

```python
def verify_reproducibility(result1, result2):
    """Check if two results came from equivalent computations."""
    return result1.hash == result2.hash
```

### Audit Trail

```python
def build_audit_trail(result):
    """Create human-readable audit trail."""
    chain = get_lineage_chain(result)
    trail = []
    for i, record in enumerate(chain):
        trail.append(f"Step {i+1}: {record.function_name}")
        for const in record.constants:
            trail.append(f"  - {const['name']}: {const['value_repr']}")
    return "\n".join(trail)
```

### Dependency Analysis

```python
def find_all_sources(result):
    """Find all leaf nodes (original data sources)."""
    chain = get_lineage_chain(result)
    sources = []
    for record in chain:
        if not record.inputs:  # No thunk inputs = source
            sources.append(record)
    return sources
```
