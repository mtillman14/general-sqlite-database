# SciStack

## Better Research Tools, Better Research Outcomes

SciStack is a database framework purpose-built for scientific data analysis. It gives you a structured, versioned, and queryable home for every piece of data your pipeline produces — from raw signals to final results — with near zero infrastructure code on your part, and **zero changes to your analysis code**.

It works in both **Python** and **MATLAB**.

## The Problem

Every scientist who writes analysis code eventually builds the same thing: a tangle of folders, naming conventions, and bookkeeping scripts to track which data came from where, which version of a function produced it, and whether it's already been computed.

That infrastructure code is never the point. But it eats weeks of your time, it's fragile, and it's different in every lab.

## Three Layers, Use What You Need

SciStack replaces all of it with three ideas:

- **Named variable types** — instead of files on disk, your data lives in typed database tables you can query by metadata
- **Automatic lineage** — a simple decorator records exactly what function and inputs produced each result
- **Computation caching** — if you've already computed something, SciStack knows and skips it

With SciStack, your analysis scripts contain _only_ analysis logic. The infrastructure is handled for you.

## Package Architecture

SciStack is a stack of libraries. You can enter at any level — each layer adds more features at the cost of a bit more setup.

```
┌────────────────────────────────────────────────────────────────────┐
│                            scihist                                 │
│  One-import entry point: for_each() with automatic DB load/save    │
│  and lineage tracking. Re-exports everything from the layers below │
│  deps: scidb + scilineage                                          │
├───────────────────────────────┬────────────────────────────────────┤
│            scidb              │            scilineage              │
│  Typed variable storage;      │  Wraps any function to record its  │
│  configure_database(),        │  full computational lineage.       │
│  for_each() that loads from   │  Enables caching and provenance    │
│  DB and saves results back    │  queries — no DB required.         │
│  deps: scifor + sciduck +     │  deps: canonical-hash              │
│        canonical-hash +       │                                    │
│        path-gen               │                                    │
├───────────────────────────────┴────────────────────────────────────┤
│                            scifor                                  │
│  Batch execution on plain tables / DataFrames — iterates over      │
│  condition combinations, slices data, collects results.            │
│  No database, no tracking, no dependencies.                        │
└────────────────────────────────────────────────────────────────────┘
```

**scifor** is the foundation: a standalone batch execution engine that works with plain MATLAB tables or pandas DataFrames. There is no setup overhead — just give it a function, your data, and the experimental conditions to iterate over.

**scidb** adds typed, versioned database storage so your variables live in queryable DuckDB tables instead of scattered files on disk. It provides `configure_database()`, `BaseVariable`, and a `for_each()` that automatically loads inputs from the database and saves results back.

**scilineage** is an independent parallel track that has no database dependency. It wraps functions with `@lineage_fcn` to record exactly what function and inputs produced each result. This unlocks caching (skip re-running a computation whose inputs haven't changed) and provenance queries.

**scihist** brings everything together. Its `for_each()` automatically wraps your functions in `@lineage_fcn`, loads inputs from the database, and saves outputs back with full lineage attached. It also re-exports the entire API from the layers below, so most users can `from scihist import *` and have everything they need.

Each layer can be used independently. `scifor` is useful when your data is already in memory and you just want structured batch processing. `scilineage` can be dropped into any pipeline to add provenance without touching your storage layer. `scidb` gives you the database without requiring lineage tracking. `scihist` is the recommended starting point for new pipelines that want the full feature set.

## Quick Start

### Installation

```bash
pip install scistack
```

This pulls in all core dependencies (`sciduckdb`, `thunk`, `scipathgen`, `canonicalhash`, `scirun`).

For development (editable installs of all packages):

```bash
git clone https://github.com/mtillman14/scistack
cd scistack
./dev-install.sh
```

### One-Time Setup

Every project starts by configuring a database. You do this once.

```python
from scifor import set_schema, for_each

set_schema(["subject", "session"])

results = for_each(
    bandpass_filter,
    inputs={"signal": raw_df, "low_hz": 20},
    subject=[1, 2, 3],
    session=["pre", "post"],
)
```

`dataset_schema_keys` describes the structure of your experiment. If your data is organized by subject and session, say so — SciStack uses this to let you save and query data naturally.

results = scifor.for_each(@bandpass_filter, ...
struct('signal', raw_tbl, 'low_hz', 20), ...
subject=[1, 2, 3], session=["pre", "post"]);

````

You tell scifor which columns identify your experimental conditions (the "schema"), and it loops over every combination, filters each table to matching rows, calls your function, and collects the results into a clean output table.

scifor is standalone and has no dependencies beyond standard data structures. It can be dropped into any project.

See the [scifor README](scifor/README.md) for more.

### Layer 2: scidb — Database Storage

Wraps scifor with a database layer. Instead of working with tables you've already loaded, scidb loads inputs from a DuckDB database and saves results back. You define named variable types for each kind of data in your pipeline, and scidb gives you structured, queryable storage with metadata-based addressing.

```python
from scidb import BaseVariable, configure_database, for_each

# One-time setup
db = configure_database("experiment.duckdb", ["subject", "session"])

# Define variable types (one-liners)
class RawEMG(BaseVariable):
    pass

class FilteredEMG(BaseVariable):
    pass

class MaxActivation(BaseVariable):
    pass
````

That's it. No configuration, no serialization code. SciStack handles numpy arrays, scalars, lists, dicts, and DataFrames natively.

### Save and Load Data

```python
import numpy as np
RawEMG.save(np.random.randn(1000), subject=1, session="pre")

# Load it back
raw = RawEMG.load(subject=1, session="pre")
print(raw.data)  # your numpy array

# Batch processing — loads from DB, runs function, saves results
for_each(
    bandpass_filter,
    inputs={"signal": RawEMG, "low_hz": 20},
    outputs=[FilteredEMG],
    subject=[1, 2, 3],
    session=["pre", "post"],
)
```

````matlab
scidb.configure_database("experiment.duckdb", ["subject", "session"]);

RawEMG().save(randn(1000, 1), subject=1, session="pre");
raw = RawEMG().load(subject=1, session="pre");

Wrap your analysis functions with `@lineage_fcn` and SciStack records which functions produced what **and the input variable values** — automatically:

```python
from scilineage import lineage_fcn

@lineage_fcn
def bandpass_filter(signal, low_hz, high_hz):
    # your filtering logic
    return filtered_signal

# Run the pipeline — lineage is tracked automatically
raw = RawEMG.load(subject=1, session="pre")
filtered = bandpass_filter(raw, low_hz=20, high_hz=450)
FilteredEMG.save(filtered, subject=1, session="pre")

# Later: "What function produced this?"
provenance = db.get_provenance(FilteredEMG, subject=1, session="pre")
print(provenance["function_name"])  # "bandpass_filter"
print(provenance["constants"])      # {"low_hz": 20, "high_hz": 450}

# Run the same pipeline again — cache hit, no recomputation
filtered = bandpass_filter(raw, low_hz=20, high_hz=450)  # returns instantly
````

Your functions stay clean, no boilerplate required. They receive normal numpy arrays and return normal values. The `@lineage_fcn` decorator handles all the bookkeeping at the boundary.

If the `@lineage_fcn` decorator is still too close to your code for your test, wrap it in a `Thunk()` call later on:

```python

from scidb.thunk import Thunk

compute_max = Thunk(compute_max)
```

**Run the same pipeline again and every step is skipped** — SciStack recognizes the same function + same inputs and returns the cached result instantly.

## Scaling Up with `for_each()`

Real experiments can have dozens of subjects and conditions, or thousands. SciStack can handle it all, using `for_each()` runs your pipeline over every combination automatically:

```python
from scidb import for_each

# 5 subjects
for_each(
    bandpass_filter,
    inputs={"signal": RawEMG},
    outputs=[FilteredEMG],
    subject=[1, 2, 3, 4, 5],
    session=["baseline", "post"],
)

# 10,000 subjects
for_each(
    bandpass_filter,
    inputs={"signal": RawEMG},
    outputs=[FilteredEMG],
    subject=range(1,10000),
    session=["baseline", "post"],
)

# Specify subject list of any size
subject_list = config["subjects"] # Load from some configuration file
for_each(
    bandpass_filter,
    inputs={"signal": RawEMG},
    outputs=[FilteredEMG],
    subject=subject_list,
    session=["baseline", "post"],
)
```

This loads `RawEMG` for each subject/session combination, runs `bandpass_filter`, and saves the result as `FilteredEMG` — multiple iterations, zero boilerplate. If a subject is missing data, that iteration is skipped gracefully. In the future, logging support is planned to document what ran successfully and what failed, and why.

Need one input to stay fixed while others iterate? Use `Fixed`:

```python
from scidb import Fixed

for_each(
    compare_to_baseline,
    inputs={
        "baseline": Fixed(RawEMG, session="baseline"),  # always load baseline
        "current": RawEMG,                               # iterates normally
    },
    outputs=[Delta],
    subject=[1, 2, 3, 4, 5],
    session=["post_1", "post_2", "post_3"],
)
```

## Powerful Querying

Because your data lives in a real database (not scattered files), querying is simple and powerful:

```python
# Load one specific record
emg = FilteredEMG.load(subject=3, session="post")

# Load all sessions for a subject — returns a list
all_sessions = FilteredEMG.load(subject=3)
for var in all_sessions:
    print(var.metadata["session"], var.data.shape)

# Load everything as a DataFrame for analysis
import pandas as pd
df = MaxActivation.load_all(as_df=True)
#   subject  session    data
#   1        baseline   0.82
#   1        post       1.47
#   2        baseline   0.91
#   ...
```

No folder traversal. No filename parsing. No `results_v2_final_FINAL.csv`. Just ask for what you want by the metadata that matters.

### Your Data Is Not Locked Away

Worried that putting data in a database means you can't see or inspect it? Don't be. SciStack uses [DuckDB](https://duckdb.org/) under the hood, and every variable type gets a human-readable **view** that you can query directly with SQL — in DBeaver, the DuckDB CLI, or any tool that speaks SQL.

For example, the `MaxActivation` view looks like this:

| subject | session  | value |
| ------- | -------- | ----- |
| 1       | baseline | 0.82  |
| 1       | post     | 1.47  |
| 2       | baseline | 0.91  |
| 2       | post     | 1.38  |
| 3       | baseline | 0.76  |
| 3       | post     | 1.22  |

You can query it directly:

```sql
SELECT subject, session, value
FROM MaxActivation;
```

Or use database viewer tools like [DBeaver](https://dbeaver.com) to view the database directly.

Your data is always one SQL query or visualization away — no Python or MATLAB required.

## Works in MATLAB Too

SciStack isn't Python-only. The entire framework works in MATLAB with a nearly identical API:

```matlab
db = scihist.configure_database("experiment.duckdb", ["subject", "session"]);

filter_fn = scidb.Thunk(@bandpass_filter);
raw = RawEMG().load(subject=1, session="pre");
filtered = filter_fn(raw, 20, 450);
FilteredEMG().save(filtered, subject=1, session="pre");
```

Your functions stay as plain functions — they receive normal arrays and return normal values. The `@lineage_fcn` decorator handles the bookkeeping at the boundary.

## What a Full Pipeline Looks Like

Here's a complete pipeline using all three layers:

```python
from scidb import BaseVariable, configure_database, for_each
from scilineage import lineage_fcn

# --- Setup ---
db = configure_database("gait_study.duckdb", ["subject", "session", "trial"])

# --- Variable types ---
class RawKinematicData(BaseVariable):
    pass

class StepLength(BaseVariable):
    pass

class MeanStepLength(BaseVariable):
    pass

# --- Processing functions ---
@lineage_fcn
def extract_step_length(kinematic_data):
    # your biomechanics logic here
    return step_lengths

@lineage_fcn
def compute_mean(values):
    return float(np.mean(values))

# --- Run the pipeline ---
for_each(
    extract_step_length,
    inputs={"kinematic_data": RawKinematicData},
    outputs=[StepLength],
    subject=[1, 2, 3],
    session=["pre", "post"],
    trial=[1, 2, 3, 4, 5],
)

for_each(
    compute_mean,
    inputs={"values": StepLength},
    outputs=[MeanStepLength],
    subject=[1, 2, 3],
    session=["pre", "post"],
)

# --- Analyze results ---
df = MeanStepLength.load_all(as_df=True)
print(df.groupby("session")["data"].mean())
```

No file I/O code. No path management. No version tracking logic.

Want to change the function logic? SciStack will automatically detect the change, and will re-run that processing step on the next run of the script. Want to change a setting to the function? SciStack will detect that too, and re-run the processing step. Data will be saved to the database, **and the previous data will be preserved**. Understanding the effect of analysis decisions on our results has never been easier.

```python
# Load one record
emg = FilteredEMG.load(subject=3, session="post")

# Load all sessions for a subject
all_sessions = FilteredEMG.load(subject=3)

# Load everything as a DataFrame
df = FilteredEMG.load_all(as_df=True)
```

```sql
SELECT subject, session, value
FROM FilteredEMG;
```

By abstracting away all infrastructure — file paths, storage formats, naming conventions — SciStack decouples your analysis logic from your local environment. Your pipeline code contains _only_ the scientific computation.

Data saved from Python can be loaded in MATLAB and vice versa. The MATLAB API mirrors the Python API closely:

Today, sharing a pipeline means sharing a pile of scripts with hardcoded paths and implicit assumptions. With SciStack, the pipeline _is_ the science, and the infrastructure adapts to wherever it runs.

## Learn More

- [Quickstart Guide](docs/quickstart.md) — Get running in 5 minutes
- [VO2 Max Walkthrough](docs/guide/walkthrough.md) — Full example pipeline with design explanations
- [Variables Guide](docs/guide/variables.md) — Deep dive into variable types
- [Lineage Guide](docs/guide/lineage.md) — How provenance tracking works
- [Batch Processing Guide](docs/guide/for_each.md) — for_each in depth
- [API Reference](docs/api.md) — Complete API documentation
