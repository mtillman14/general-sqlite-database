# Package Distribution & PyPI Publishing

## Package Names

Each sub-package is independently publishable to PyPI. Two packages were renamed to avoid PyPI conflicts:

| Directory | PyPI Name | Import Name | Notes |
|-----------|-----------|-------------|-------|
| `canonical-hash/` | `canonicalhash` | `canonicalhash` | |
| `path-gen/` | `scipathgen` | `scipathgen` | Renamed from `pathgen` (taken on PyPI) |
| `pipelinedb-lib/` | `pipelinedb` | `pipelinedb` | |
| `scirun-lib/` | `scirun` | `scirun` | |
| `sciduck/` | `sciduckdb` | `sciduckdb` | Renamed from `sciduck` (taken on PyPI) |
| `thunk-lib/` | `thunk` | `thunk` | |
| `src/` (root) | `scidb` | `scidb` | |
| `scidb-matlab/` | `scidb-matlab` | `scidb_matlab` | |
| `scidb-net/` | `scidb-net` | `scidbnet` | |

**Important**: The directory names on disk did NOT change (e.g. the folder is still `sciduck/`), only the Python package directory inside `src/` was renamed (e.g. `sciduck/src/sciduckdb/`). This means `sys.path.insert` references to `sciduck/src` in conftest files still work — they add the `src/` directory to the path, and Python finds the `sciduckdb` package inside it.

## Dependency Order

```
Layer 0 (no internal deps): canonicalhash, scipathgen, pipelinedb, scirun, sciduckdb
Layer 1:                     thunk (depends: canonicalhash)
Layer 2:                     scidb (depends: thunk, scipathgen, canonicalhash, sciduckdb, scirun)
Layer 3:                     scidb-matlab, scidb-net (depend: scidb)
```

## Build System

All packages use **hatchling** as the build backend. Previously, the root `scidb` and `scidb-matlab` used setuptools; they were migrated.

Each `pyproject.toml` has:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.sdist]
include = ["/src"]

[tool.hatch.build.targets.wheel]
packages = ["src/<package_name>"]
```

## Development Install

`dev-install.sh` at the repo root installs all packages in editable mode in dependency order. Run it inside a virtualenv:

```bash
./dev-install.sh
```

This replaces the old `sys.path.insert()` hacks that were in `src/scidb/__init__.py` and `src/scidb/database.py`. Those hacks injected sibling package `src/` directories into `sys.path` at runtime so imports would resolve without installation. With editable installs, Python's normal import machinery handles everything.

**Note**: Some conftest.py files and test files still contain `sys.path.insert` lines for the monorepo packages. These are redundant when packages are pip-installed but harmless — they exist as a fallback for running tests without `dev-install.sh`. The MATLAB `setup_paths.m` also adds Python paths this way for the MATLAB-Python bridge.

## PyPI Publishing

`.github/workflows/publish.yml` uses trusted publishing (OIDC) via `pypa/gh-action-pypi-publish`. It triggers on version tags with the pattern `<package>-v<version>`, e.g.:

- `canonicalhash-v0.1.0`
- `scidb-v1.0.0`
- `sciduckdb-v0.2.0`

The workflow maps the tag prefix to the correct directory and builds/publishes that single package.

## Key Decisions

- **Monorepo with independent PyPI packages**: Each package has its own version and can be released independently.
- **`pip install scidb`** pulls in all core dependencies automatically via the dependency list in the root `pyproject.toml`.
- **Import name = distribution name** for all packages (no mismatches to remember).
