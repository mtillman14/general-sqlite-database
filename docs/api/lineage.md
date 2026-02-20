# Lineage API — Thunk System

The thunk system provides automatic provenance tracking and computation caching. Wrap any processing function with `@thunk` (Python) or `scidb.Thunk` (MATLAB) and every call is recorded — including what inputs were used and what parameters were set. Re-running the same computation against already-saved results returns the cached result instantly.

---

## `@thunk` / `scidb.Thunk`

Wraps a function so that every call produces a `ThunkOutput` instead of raw data. The `ThunkOutput` carries both the result and its lineage — which function was called, with what inputs, and what constant values were used.

=== "Python"

    ```python
    from scidb import thunk, Thunk

    # Decorator style
    @thunk
    def bandpass_filter(signal: np.ndarray, low: float, high: float) -> np.ndarray:
        # Your filtering code here
        return filtered

    result = bandpass_filter(raw_signal, 20.0, 450.0)
    result.data       # the filtered array
    result.hash       # lineage hash (used for cache lookup)

    # Functional style (for external functions)
    from scipy.signal import filtfilt
    thunked_filtfilt = Thunk(filtfilt)
    filtered = thunked_filtfilt(b, a, signal)
    ```

=== "MATLAB"

    ```matlab
    % Wrap a named function (must be a named function, not anonymous)
    filter_fn = scidb.Thunk(@bandpass_filter);
    result = filter_fn(raw_signal, 20, 450);
    result.data       % the filtered array

    % Multi-output: function must return a cell array
    split_fn = scidb.Thunk(@split_data, unpack_output=true);
    [first, second] = split_fn(data);
    ```

**Key behavior:**

- Inputs that are `BaseVariable` instances or `ThunkOutput`s are automatically **unwrapped** to raw data before calling the function. Your function receives plain arrays — not framework wrapper types.
- The lineage chain is preserved: if `result` came from another thunk, that dependency is recorded.
- Lineage is only stored in the database when you call `save()` on the result.

---

## `@thunk` Options

### `unpack_output`

When your function returns a tuple/cell array that should be split into separate outputs:

=== "Python"

    ```python
    @thunk(unpack_output=True)
    def design_filter(order, cutoff, fs):
        b, a = scipy.signal.butter(order, cutoff, fs=fs)
        return b, a  # returned as a tuple

    b, a = design_filter(4, 20, 1000)
    # b and a are separate ThunkOutputs with the same lineage parent
    ```

=== "MATLAB"

    ```matlab
    design_fn = scidb.Thunk(@design_filter, unpack_output=true);
    % Function must return a cell array: {b, a}
    [b, a] = design_fn(4, 20, 1000);
    ```

### `unwrap=False` (Python only)

By default, thunks unwrap `BaseVariable` / `ThunkOutput` inputs to raw data. Pass `unwrap=False` to receive the wrapper objects inside your function — useful for accessing `record_id` or `metadata` during debugging:

```python
@thunk(unwrap=False)
def debug_process(var):
    # var is a BaseVariable, not raw data
    print(f"Input record_id: {var.record_id}")
    print(f"Metadata: {var.metadata}")
    return var.data * 2
```

### `generates_file`

For pipeline steps that produce files (plots, reports, exports) rather than data to store in the database. The thunk runs once and is skipped on subsequent runs when inputs haven't changed.

=== "Python"

    ```python
    @thunk(generates_file=True)
    def plot_signal(data, subject, session):
        plt.figure()
        plt.plot(data)
        plt.savefig(f"signal_s{subject}_{session}.png")

    # Use a "sentinel" variable to record that this ran
    class Figure(BaseVariable):
        pass

    data = ProcessedData.load(subject=1, session="A")
    result = plot_signal(data, subject=1, session="A")
    Figure.save(result, subject=1, session="A")
    # Second run: result.data is None, no re-execution

    # With for_each: metadata is automatically passed as kwargs
    for_each(
        plot_signal,
        inputs={"data": ProcessedData},
        outputs=[Figure],
        subject=[1, 2, 3],
        session=["A", "B"],
    )
    ```

=== "MATLAB"

    Equivalent behavior is available; the `generates_file` attribute is set on the Python-side Thunk proxy. Contact your team's Python developer to configure this for shared workflows.

---

## Wrapping External Functions

Use `Thunk(fn)` to add lineage tracking to functions from external libraries:

=== "Python"

    ```python
    from scidb import Thunk
    from scipy.signal import butter, filtfilt, welch
    from sklearn.decomposition import PCA

    # unpack_output=True for functions that return tuples
    thunked_butter   = Thunk(butter, unpack_output=True)
    thunked_filtfilt = Thunk(filtfilt)
    thunked_welch    = Thunk(welch, unpack_output=True)
    thunked_pca      = Thunk(PCA(n_components=10).fit_transform)

    # Use exactly like normal scipy functions
    b, a = thunked_butter(N=4, Wn=0.1, btype='low')
    filtered = thunked_filtfilt(b, a, raw_signal)
    freqs, psd = thunked_welch(filtered, fs=1000)
    ```

=== "MATLAB"

    ```matlab
    % MATLAB: wrap any named function
    filtfilt_fn = scidb.Thunk(@filtfilt);
    filtered = filtfilt_fn(b, a, raw_signal);
    ```

---

## `ThunkOutput`

The return value from any thunk call. You typically pass it directly to `save()` — but you can also inspect it.

=== "Python"

    | Attribute | Type | Description |
    |-----------|------|-------------|
    | `data` | any | The computed result |
    | `hash` | `str` | Lineage hash (used for cache lookup) |
    | `pipeline_thunk` | `PipelineThunk` | The invocation record with captured inputs |
    | `is_complete` | `bool` | Whether the computation actually ran (False on cache hit for `generates_file`) |

    ```python
    result = bandpass_filter(raw, 20, 450)
    print(result.data)   # the filtered array
    FilteredSignal.save(result, subject=1, session="A")
    ```

=== "MATLAB"

    | Property | Type | Description |
    |----------|------|-------------|
    | `data` | MATLAB type | The computed result |
    | `record_id` | `string` | Set after save/load; empty before |
    | `metadata` | `struct` | Set after load |
    | `content_hash` | `string` | Hash of data content |
    | `lineage_hash` | `string` | Lineage hash; empty for raw data |

    ```matlab
    result = filter_fn(raw_signal, 20, 450);
    disp(result.data);       % the filtered array
    FilteredSignal().save(result, subject=1, session="A");
    ```

---

## How Caching Works

After `save()`, the lineage hash is recorded in the SQLite database. On subsequent runs:

1. The thunk computes its lineage hash from the function's bytecode hash + input content hashes
2. It checks the database for a matching hash
3. If found, it loads and returns the previously saved result — **without executing the function**

```
First run:  bandpass_filter(raw, 20, 450)  →  executes  →  save()  →  cached
Second run: bandpass_filter(raw, 20, 450)  →  cache hit  →  returns saved result
```

The cache is invalidated automatically when:

- The function's code changes (bytecode hash changes)
- The input data changes (input record_id or content hash changes)
- A constant argument changes (e.g., `20` → `25`)

For multi-output functions, **all** outputs must be saved before caching takes effect.

---

## Cross-Script Lineage

Lineage is preserved when data is saved to the database and then loaded in a separate script or session:

=== "Python"

    ```python
    # step1.py
    @thunk
    def preprocess(data):
        return data * 2

    result = preprocess(raw_data)
    Intermediate.save(result, subject=1)

    # step2.py — separate execution
    loaded = Intermediate.load(subject=1)

    @thunk
    def analyze(data):   # receives the raw array, unwrapped from loaded
        return data.mean()

    # Pass the loaded variable (not loaded.data) to preserve lineage
    final = analyze(loaded)
    FinalResult.save(final, subject=1)
    # Lineage: FinalResult ← analyze ← Intermediate ← preprocess
    ```

=== "MATLAB"

    ```matlab
    % Step 1
    preprocess_fn = scidb.Thunk(@preprocess);
    result = preprocess_fn(raw_data);
    Intermediate().save(result, subject=1);

    % Step 2 (separate session)
    loaded = Intermediate().load(subject=1);
    analyze_fn = scidb.Thunk(@analyze);
    final = analyze_fn(loaded);   % pass the loaded variable, not loaded.data
    FinalResult().save(final, subject=1);
    ```

---

## Manual Lineage Inspection (Python)

For inspecting lineage without saving:

```python
from scidb.lineage import extract_lineage, get_raw_value

result = bandpass_filter(raw, 20, 450)

lineage = extract_lineage(result)
print(lineage.function_name)   # "bandpass_filter"
print(lineage.inputs)          # list of input descriptors
print(lineage.constants)       # [{"name": "low", "value": 20}, ...]

raw_value = get_raw_value(result)  # extracts .data if ThunkOutput, passes through otherwise
```

---

## Function Hashing

Functions are identified by a hash of their bytecode and constants. This means:

| Change | Cache effect |
|--------|-------------|
| Edit function body | Cache invalidated (bytecode changes) |
| Change a hardcoded constant (`x * 2` → `x * 3`) | Cache invalidated |
| Rename a local variable | No effect (bytecode unchanged) |
| Add a comment | No effect |
| Change function argument value at call site | Cache invalidated (inputs change) |

In MATLAB, the entire source file is hashed — any edit to any function in the same `.m` file will invalidate the cache for that function.
