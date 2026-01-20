# Lineage Tracking

SciDB automatically tracks data provenance using the `@thunk` decorator. When you save data produced by a thunked function, the lineage is captured automatically.

## The `@thunk` Decorator

Wrap processing functions with `@thunk` to enable lineage tracking:

```python
from scidb import thunk

@thunk(n_outputs=1)
def process_signal(signal: np.ndarray, factor: float) -> np.ndarray:
    return signal * factor

# Returns OutputThunk, not raw array
result = process_signal(data, 2.5)
print(result.data)  # The actual array
```

### Multiple Outputs

```python
@thunk(n_outputs=2)
def split_data(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mid = len(data) // 2
    return data[:mid], data[mid:]

first_half, second_half = split_data(data)
```

## How It Works

1. **Thunk wraps function** - Creates a `Thunk` object with function hash
2. **Call creates PipelineThunk** - Captures all inputs
3. **Execution returns OutputThunk** - Wraps result with lineage reference
4. **Save extracts lineage** - Stores provenance in database

```
@thunk decorator
    │
    ▼
┌─────────┐      ┌───────────────┐      ┌─────────────┐
│  Thunk  │──────│ PipelineThunk │──────│ OutputThunk │
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

When saving an `OutputThunk`, lineage is captured automatically:

```python
@thunk(n_outputs=1)
def normalize(arr):
    return (arr - arr.mean()) / arr.std()

raw = RawData.load(subject=1)
normalized = normalize(raw.data)

# Lineage captured on save
NormalizedData.save(normalized, subject=1, stage="normalized")
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

### Full Lineage Chain

To see the complete computation history (not just immediate parent):

```python
# Get nested dict with full lineage
lineage = db.get_full_lineage(FinalResult, subject=1)
print(lineage)
# {
#     "type": "FinalResult",
#     "record_id": "abc123...",
#     "function": "compute_stats",
#     "function_hash": "...",
#     "constants": [...],
#     "inputs": [
#         {
#             "type": "NormalizedData",
#             "record_id": "...",
#             "function": "normalize",
#             "inputs": [...]
#         }
#     ]
# }

# Get print-friendly version
print(db.format_lineage(FinalResult, subject=1))
```

Output of `format_lineage()`:
```
└── FinalResult (record_id: abc123...)
    ├── function: compute_stats [hash: def456...]
    ├── constants: threshold=0.5
    └── inputs:
        └── NormalizedData (record_id: ghi789...)
            ├── function: normalize [hash: jkl012...]
            └── inputs:
                └── RawData (record_id: mno345...)
                    └── [source: manual]
```

### What Variables Derived From This?

```python
derived = db.get_derived_from(RawData, subject=1)

for d in derived:
    print(f"{d['type']} via {d['function']}")
# Output: "NormalizedData via normalize"
```

## Chained Pipelines

Lineage tracks through multiple processing steps:

```python
@thunk(n_outputs=1)
def step1(data):
    return data * 2

@thunk(n_outputs=1)
def step2(data):
    return data + 1

@thunk(n_outputs=1)
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
from scidb import extract_lineage, get_raw_value

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

thunked_butter = Thunk(butter, n_outputs=2)      # Returns (b, a)
thunked_filtfilt = Thunk(filtfilt, n_outputs=1)
thunked_welch = Thunk(welch, n_outputs=2)        # Returns (freqs, psd)
```

### Example: Signal Processing Pipeline

```python
from scipy.signal import butter, filtfilt, welch
from scidb import Thunk, BaseVariable, configure_database
import numpy as np

# Wrap scipy functions
thunked_butter = Thunk(butter, n_outputs=2)
thunked_filtfilt = Thunk(filtfilt, n_outputs=1)
thunked_welch = Thunk(welch, n_outputs=2)

# Define variable types
class SignalData(BaseVariable):
    schema_version = 1
    def to_db(self):
        return pd.DataFrame({"value": self.data})
    @classmethod
    def from_db(cls, df):
        return df["value"].values

class PSDData(BaseVariable):
    schema_version = 1
    def to_db(self):
        return pd.DataFrame({"freq": self.data[0], "power": self.data[1]})
    @classmethod
    def from_db(cls, df):
        return (df["freq"].values, df["power"].values)

# Setup
db = configure_database("experiment.db")

# Run pipeline with full lineage tracking
raw_signal = SignalData.load(subject=1, session="baseline")

b, a = thunked_butter(N=4, Wn=[1, 40], btype='band', fs=1000)
filtered = thunked_filtfilt(b.data, a.data, raw_signal.data)
freqs, psd = thunked_welch(filtered, fs=1000)

# Save with lineage
SignalData.save(filtered, subject=1, session="filtered")
PSDData.save((freqs, psd), subject=1, session="psd")

# View full lineage
print(db.format_lineage(PSDData, subject=1, session="psd"))
```

### Example: Machine Learning Pipeline

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scidb import Thunk

# Wrap sklearn - note: wrap the methods, not the class
scaler = StandardScaler()
pca = PCA(n_components=10)

thunked_fit_transform = Thunk(scaler.fit_transform, n_outputs=1)
thunked_pca_fit_transform = Thunk(pca.fit_transform, n_outputs=1)

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
thunk_butter = Thunk(butter, n_outputs=2)
thunk_filtfilt = Thunk(filtfilt, n_outputs=1)
thunk_welch = Thunk(welch, n_outputs=2)
thunk_hilbert = Thunk(hilbert, n_outputs=1)
thunk_fft = Thunk(fft, n_outputs=1)
thunk_ifft = Thunk(ifft, n_outputs=1)
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
@thunk(n_outputs=1)
def preprocess(data):
    return data * 2

result = preprocess(raw_data)
Intermediate.save(result, db=db, subject=1, stage="preprocessed")
```

```python
# step2.py (separate execution)
loaded = Intermediate.load(db=db, subject=1, stage="preprocessed")

@thunk(n_outputs=1)
def analyze(data):
    # Receives the raw numpy array (loaded.data)
    return data.mean()

# Pass the loaded variable - lineage links to loaded.record_id
result = analyze(loaded)
FinalResult.save(result, db=db, subject=1, stage="analyzed")

# Lineage correctly shows: FinalResult <- analyze <- Intermediate
```

The key: pass the `BaseVariable` instance, not `loaded.data`. The thunk automatically unwraps it.

## Debugging with `unwrap=False`

By default, thunks unwrap `BaseVariable` and `OutputThunk` inputs to their raw data. Use `unwrap=False` to receive the wrapper objects for debugging:

```python
@thunk(n_outputs=1, unwrap=False)
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
@thunk(n_outputs=1)
def process_item(item):
    return item * 2

results = []
for item in items:
    results.append(process_item(item))  # Returns OutputThunk
pd.concat(results)  # Error: can't concat OutputThunks

# DO this instead
def process_item(item):  # No @thunk
    return item * 2

@thunk(n_outputs=1)
def process_all(items):
    return [process_item(item) for item in items]
```
