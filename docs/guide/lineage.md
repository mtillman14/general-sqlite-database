# Lineage Tracking

SciDB automatically tracks data provenance using the `@thunk` decorator. When you save data produced by a thunked function, the lineage is captured automatically.

## The `@thunk` Decorator

Wrap processing functions with `@thunk` to enable lineage tracking:

```python
from scidb import thunk

@thunk
def process_signal(signal: np.ndarray, factor: float) -> np.ndarray:
    return signal * factor

# Returns ThunkOutput, not raw array
result = process_signal(data, 2.5)
print(result.data)  # The actual array
```

### Multiple Outputs

```python
@thunk(unpack_output=True)
def split_data(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mid = len(data) // 2
    return data[:mid], data[mid:]

first_half, second_half = split_data(data)
```

## How It Works

1. **Thunk wraps function** - Creates a `Thunk` object with function hash
2. **Call creates PipelineThunk** - Captures all inputs
3. **Execution returns ThunkOutput** - Wraps result with lineage reference
4. **Save extracts lineage** - Stores provenance in PipelineDB (SQLite)

```
@thunk decorator
    │
    ▼
┌─────────┐      ┌───────────────┐      ┌─────────────┐
│  Thunk  │──────│ PipelineThunk │──────│ ThunkOutput │
│ (func)  │      │ (inputs)      │      │ (result)    │
└─────────┘      └───────────────┘      └─────────────┘
    │                   │                      │
    │                   │                      ▼
    │                   │               ┌─────────────┐
    │                   └──────────────▶│  .save()    │
    │                                   │  captures   │
    └──────────────────────────────────▶│  lineage    │
                                        └─────────────┘
```

## Automatic Lineage Capture

When saving a `ThunkOutput`, lineage is captured automatically:

```python
@thunk
def normalize(arr):
    return (arr - arr.mean()) / arr.std()

raw = RawData.load(subject=1)
normalized = normalize(raw)  # Pass the variable, not .data

# Lineage captured on save
NormalizedData.save(normalized, subject=1, stage="normalized")
```

**Important:** Pass the `BaseVariable` instance (not `.data`) to the thunk to preserve lineage tracking.

## How Lineage Is Stored

Lineage is stored in a separate SQLite database (PipelineDB) as **global computational
provenance**. Each lineage record maps an output `record_id` to the function and inputs
that produced it. The lineage table has this structure:

| Column | Description |
|---|---|
| `output_record_id` | The `record_id` of the saved output |
| `output_type` | Variable class name (e.g., `"NormalizedData"`) |
| `lineage_hash` | Hash of the full computation (function + inputs), used for cache lookups |
| `function_name` | Name of the function that produced the output |
| `function_hash` | SHA-256 hash of the function's bytecode |
| `inputs` | JSON list of input descriptors (see below) |
| `constants` | JSON list of constant value descriptors |

### Lineage is not schema-aware

The lineage table tracks relationships between `record_id` values. It does **not** have
schema key columns (e.g., `subject`, `session`). This means:

- Lineage records are **global** — they are not scoped to any particular schema location.
- You cannot directly query "show me all computations for subject=1" from the lineage
  table alone.
- To answer schema-scoped provenance questions (e.g., "what inputs at subject=1, session=1
  produced this output?"), you need to:
  1. Look up the output's `record_id` by schema keys in the data table (DuckDB)
  2. Query the lineage table by that `record_id` (PipelineDB)
  3. For each input `record_id` in the lineage record, look up its schema location
     in the data table (DuckDB)

The input descriptors for saved variables do include a `metadata` field with the
original schema keys, but this is stored as an opaque JSON blob inside the `inputs`
column — it is not indexed or directly queryable.

```
Global provenance (what lineage stores):
    record abc123 ──[normalize()]──> record def456

Schema-scoped provenance (requires joining data + lineage tables):
    RawData(subject=1, session=1, record=abc123)
        ──[normalize(), hash=a1b2c3]──>
    NormalizedData(subject=1, session=1, record=def456)
```

## Querying Provenance

### What Produced This Variable?

```python
provenance = db.get_provenance(NormalizedData, subject=1, stage="normalized")

print(provenance["function_name"])   # "normalize"
print(provenance["function_hash"])   # SHA-256 of function bytecode
print(provenance["inputs"])          # List of input descriptors
print(provenance["constants"])       # List of constant values
```

Note: `get_provenance()` handles the data-table-to-lineage-table join for you.
It first looks up the `record_id` matching the given metadata, then queries the
lineage table. The returned `inputs` list contains `record_id` references to
upstream variables, but not their schema locations — use `db.load()` with the
input `record_id` to resolve where each input lives in the dataset.

### Check Lineage Exists

```python
if db.has_lineage(record_id):
    print("This variable was produced by a thunked function")
```

## Chained Pipelines

Lineage tracks through multiple processing steps:

```python
@thunk
def step1(data):
    return data * 2

@thunk
def step2(data):
    return data + 1

@thunk
def step3(data):
    return data ** 2

# Chain of operations
result = step3(step2(step1(raw_data)))

# Lineage captures full chain
MyVar.save(result, subject=1, stage="final")
```

## Manual Lineage Extraction

For inspection without saving:

```python
from scidb.lineage import extract_lineage, get_raw_value

result = process(data)

# Extract lineage record
lineage = extract_lineage(result)
print(lineage.function_name)
print(lineage.inputs)

# Get raw value
raw_value = get_raw_value(result)
```

## Function Hashing

Functions are identified by a SHA-256 hash of:

- Bytecode (`__code__.co_code`)
- Constants (`__code__.co_consts`)

This means:

- Same function logic = same hash (reproducible)
- Different constants = different hash (e.g., `x * 2` vs `x * 3`)
- Renamed variables don't change the hash

## Wrapping External Functions

One of the main goals of scientific workflows is leveraging existing libraries. You can wrap any external function as a `Thunk` to get lineage tracking:

```python
from scidb import Thunk

# Wrap functions from any package
from scipy.signal import butter, filtfilt, welch
from sklearn.preprocessing import StandardScaler

thunked_butter = Thunk(butter)      # Returns (b, a)
thunked_filtfilt = Thunk(filtfilt)
thunked_welch = Thunk(welch)        # Returns (freqs, psd)
```

### Example: Signal Processing Pipeline

```python
from scipy.signal import butter, filtfilt, welch
from scidb import Thunk, BaseVariable, configure_database
import numpy as np

# Wrap scipy functions
thunked_butter = Thunk(butter)
thunked_filtfilt = Thunk(filtfilt)
thunked_welch = Thunk(welch)

# Define variable types (native storage)
class SignalData(BaseVariable):
    schema_version = 1

class PSDData(BaseVariable):
    schema_version = 1

# Setup
db = configure_database("experiment.duckdb", ["subject", "session"], "pipeline.db")

# Run pipeline with full lineage tracking
raw_signal = SignalData.load(subject=1, session="baseline")

b, a = thunked_butter(N=4, Wn=[1, 40], btype='band', fs=1000)
filtered = thunked_filtfilt(b.data, a.data, raw_signal.data)
freqs, psd = thunked_welch(filtered, fs=1000)

# Save with lineage
SignalData.save(filtered, subject=1, session="filtered")
PSDData.save((freqs, psd), subject=1, session="psd")
```

### Example: Machine Learning Pipeline

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scidb import Thunk

# Wrap sklearn - note: wrap the methods, not the class
scaler = StandardScaler()
pca = PCA(n_components=10)

thunked_fit_transform = Thunk(scaler.fit_transform)
thunked_pca_fit_transform = Thunk(pca.fit_transform)

# Pipeline with lineage
scaled = thunked_fit_transform(raw_features)
reduced = thunked_pca_fit_transform(scaled)

ReducedFeatures.save(reduced, subject=1, stage="pca")
```

### Creating a Thunk Library

For frequently used external functions, create a module of pre-wrapped thunks:

```python
# my_project/thunks.py
from scidb import Thunk
from scipy.signal import butter, filtfilt, welch, hilbert
from scipy.fft import fft, ifft

# Signal processing
thunk_butter = Thunk(butter)
thunk_filtfilt = Thunk(filtfilt)
thunk_welch = Thunk(welch)
thunk_hilbert = Thunk(hilbert)
thunk_fft = Thunk(fft)
thunk_ifft = Thunk(ifft)
```

```python
# In your pipeline
from my_project.thunks import thunk_filtfilt, thunk_welch

filtered = thunk_filtfilt(b, a, data)
freqs, psd = thunk_welch(filtered)
```

## Cross-Script Lineage

When pipelines are split across separate files/scripts, lineage is still tracked by passing the loaded variable to the thunk:

```python
# step1.py
@thunk
def preprocess(data):
    return data * 2

result = preprocess(raw_data)
Intermediate.save(result, subject=1, stage="preprocessed")
```

```python
# step2.py (separate execution)
loaded = Intermediate.load(subject=1, stage="preprocessed")

@thunk
def analyze(data):
    # Receives the raw numpy array (unwrapped from loaded)
    return data.mean()

# Pass the loaded variable - lineage links to loaded.record_id
result = analyze(loaded)
FinalResult.save(result, subject=1, stage="analyzed")

# Lineage correctly shows: FinalResult <- analyze <- Intermediate
```

The key: pass the `BaseVariable` instance, not `loaded.data`. The thunk automatically unwraps it.

## Debugging with `unwrap=False`

By default, thunks unwrap `BaseVariable` and `ThunkOutput` inputs to their raw data. Use `unwrap=False` to receive the wrapper objects for debugging:

```python
@thunk(unwrap=False)
def debug_process(var):
    # var is the BaseVariable, not raw data
    print(f"Input record_id: {var.record_id}")
    print(f"Input metadata: {var.metadata}")
    print(f"Data shape: {var.data.shape}")
    return var.data * 2

# Lineage still captured, but function can inspect metadata
result = debug_process(loaded)
```

This is useful for:

- Tracing data provenance during debugging
- Logging metadata alongside processing
- Building introspection tools

## Limitations

### Functions in Loops

Don't thunk functions called in loops that accumulate results:

```python
# DON'T do this
@thunk
def process_item(item):
    return item * 2

results = []
for item in items:
    results.append(process_item(item))  # Returns ThunkOutput
pd.concat(results)  # Error: can't concat ThunkOutputs

# DO this instead
def process_item(item):  # No @thunk
    return item * 2

@thunk
def process_all(items):
    return [process_item(item) for item in items]
```
