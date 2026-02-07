# Canonical Hash

Deterministic hashing for arbitrary Python objects.

Provides utilities for creating stable, deterministic hashes of Python objects, essential for cache key computation, data versioning, and reproducibility.

## Usage

```python
from canonicalhash import canonical_hash, generate_record_id

# Hash any supported Python object
h = canonical_hash(42)
h = canonical_hash([1, 2, 3])
h = canonical_hash({"key": "value"})

# Generates a 16-character hex string (first 64 bits of SHA-256)
assert len(h) == 16
assert canonical_hash(42) == canonical_hash(42)  # Deterministic
```

## Supported Types

1. JSON-serializable primitives (None, bool, int, float, str)
2. numpy ndarrays (via shape + dtype + raw bytes)
3. pandas DataFrames (via columns + index + array serialization)
4. pandas Series (via name + array serialization)
5. Dicts (sorted keys, recursive serialization)
6. Lists/tuples (order-preserving, recursive serialization)

## `generate_record_id`

Generate a unique record ID from type, schema version, content hash, and metadata:

```python
rid = generate_record_id("MyData", 1, "abc123", {"subject": 1})
```
