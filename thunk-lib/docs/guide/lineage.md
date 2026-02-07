# Lineage Tracking

Thunk automatically tracks the computational lineage of every result, enabling full reproducibility and provenance analysis.

## Understanding Lineage

When you chain thunked functions, each `ThunkOutput` maintains a reference to:

1. The `PipelineThunk` that created it (which function, which inputs)
2. Any `ThunkOutput` inputs (creating a graph of computations)

```python
from thunk import thunk

@thunk
def load(path):
    return [1, 2, 3]  # Simulated data loading

@thunk
def normalize(data):
    max_val = max(data)
    return [x / max_val for x in data]

@thunk
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

## Full Upstream Lineage

To traverse the entire computation history, use `get_upstream_lineage()`. This returns a list of dicts (one per upstream computation):

```python
from thunk import get_upstream_lineage

chain = get_upstream_lineage(scaled)

for record in chain:
    print(f"{record['function_name']}:")
    print(f"  Inputs: {len(record['inputs'])}")
    print(f"  Constants: {len(record['constants'])}")

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

## Finding Unsaved Variables

Use `find_unsaved_variables()` to find any `BaseVariable` in the upstream chain that hasn't been saved (has no `record_id`). This is used by strict lineage mode to validate that all intermediates are saved:

```python
from thunk import find_unsaved_variables

unsaved = find_unsaved_variables(result)
for variable, path in unsaved:
    print(f"Unsaved: {type(variable).__name__} at {path}")
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
from thunk import get_upstream_lineage

def build_audit_trail(result):
    """Create human-readable audit trail."""
    chain = get_upstream_lineage(result)
    trail = []
    for i, record in enumerate(chain):
        trail.append(f"Step {i+1}: {record['function_name']}")
        for const in record['constants']:
            trail.append(f"  - {const['name']}: {const['value_repr']}")
    return "\n".join(trail)
```

### Dependency Analysis

```python
from thunk import get_upstream_lineage

def find_all_sources(result):
    """Find all leaf nodes (original data sources)."""
    chain = get_upstream_lineage(result)
    sources = []
    for record in chain:
        if not record['inputs']:  # No thunk inputs = source
            sources.append(record)
    return sources
```
