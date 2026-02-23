# Path Gen

Template-based path generation for data pipelines.

Generates file paths with associated metadata from a template and metadata value combinations.

## Usage

```python
from scipathgen import PathGenerator

paths = PathGenerator(
    "{subject}/trial_{trial}.mat",
    root_folder="/data/experiment",
    subject=range(3),
    trial=range(5),
)

for path, meta in paths:
    print(path, meta)
# /data/experiment/0/trial_0.mat {'subject': 0, 'trial': 0}
# /data/experiment/0/trial_1.mat {'subject': 0, 'trial': 1}
# ...

# Supports indexing and length
print(len(paths))  # 15
path, meta = paths[0]
```

If `root_folder` is omitted, paths are resolved relative to the current working directory via `Path.resolve()`.
