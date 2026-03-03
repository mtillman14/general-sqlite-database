# SciRun

Batch execution utilities for data pipelines.

Provides utilities for running functions over combinations of metadata, automatically loading inputs and saving outputs.

## Usage

```python
from scirun import for_each, Fixed

for_each(
    process_data,
    inputs={"raw": RawData, "calibration": Fixed(Calibration, session="baseline")},
    outputs=[ProcessedData],
    subject=[1, 2, 3],
    session=["A", "B", "C"],
)
```

### `for_each`

Executes a function for all combinations of metadata, loading inputs and saving outputs automatically.

### `Fixed`

Wrapper to specify fixed metadata overrides for an input. Use when an input should be loaded with different metadata than the current iteration.

```python
# Always load baseline from session="BL", regardless of current session
for_each(
    compare_to_baseline,
    inputs={
        "baseline": Fixed(StepLength, session="BL"),
        "current": StepLength,
    },
    outputs=[Delta],
    subject=[1, 2, 3],
    session=["A", "B", "C"],
)
```
