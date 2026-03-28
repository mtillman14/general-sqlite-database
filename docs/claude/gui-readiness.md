# GUI Readiness Assessment

This document captures what abstractions are already in place and what is missing for building a GUI on top of SciStack. The GUI concept is a git-style branching map of the pipeline, usable at any layer from scifor through scihist.

## The GUI Vision

The core GUI concept is a **git-style branching pipeline map**:

- Starts as a linear graph: variable types as nodes, functions as edges
- Branches whenever different versions of a variable are generated (e.g., a parameter sweep of `low_hz=20` vs `low_hz=30`)
- Each branch is a unique path through the pipeline defined by a unique parameter/input combination
- Users can browse data, inspect provenance, view version history, and trigger pipeline runs

---

## What's Already in Place

### Data browsing
- `load()`, `load_all()`, `head()` with `where=` filters — a data browser can be built on these today
- `load_all(as_df=True)` provides tabular previews

### Schema and data discovery
- `db.dataset_schema_keys` — exposes the experiment schema (e.g., `["subject", "session"]`)
- `db.distinct_schema_values(key)` — enumerates all values for a schema key
- `db.distinct_schema_combinations(keys)` — enumerates all existing schema combinations

### Provenance queries
- `db.get_provenance(VariableClass, **metadata)` — one-hop "what function produced this?" query
- `db.get_provenance_by_schema(**schema_keys)` — all lineage records at a schema location
- `db.get_pipeline_structure()` — unique `(function, input_types, output_type)` combinations; the skeleton of a pipeline graph

### Version history
- `list_versions(**metadata)` — all stored versions at a schema location

### Manual intervention
- `scilineage.manual(data, label, reason)` — re-enters the pipeline after a user edit, documenting the intervention in lineage; conceptually hookable from a GUI

---

## What's Missing

### 1. Variable type discovery from DB ✓

`db.list_variables()` queries the `_variables` table and returns a DataFrame with `variable_name`, `schema_level`, `created_at`, `description` — one row per variable type stored in the database. This works without the original Python class definitions.

---

### 2. Constants not tracked in version keys → branch collision ✓ IMPLEMENTED

`ForEachConfig._get_direct_constants()` now extracts scalar constants and `to_version_keys()` serializes them under `__constants`. Two `for_each` calls that differ only in a constant now produce records with **distinct version keys**, so both are stored and discoverable.

Additionally, a `branch_params` column in `_record_metadata` accumulates all branch-discriminating parameters (both scalar constants, namespaced by function, and dynamic input discriminators) for each record. This enables the load API described under item 3.

---

### 3. No load API for branch filtering ✓ IMPLEMENTED

**What was added:**
- `branch_params` column in `_record_metadata`: flat JSON dict accumulating all discriminating params (constants namespaced as `fn.param`, dynamic input columns bare)
- `load(**branch_params)`: non-schema kwargs are treated as branch_params filters; `AmbiguousVersionError` is raised when multiple variants exist and no filter narrows to one
- `list_versions(...)`: returns `branch_params` per record; accepts non-schema kwargs as branch_params filters; `include_excluded=True` shows excluded variants
- `db.exclude_variant(record_id_or_type, **kwargs)` / `db.include_variant(...)`: mark variants as excluded from default queries
- `AmbiguousVersionError`, `AmbiguousParamError`: new exceptions exported from `scidb`
- Suffix matching: `load(low_hz=20)` matches `bandpass_filter.low_hz=20` in branch_params

**Also added:** `db.list_pipeline_variants()` — returns distinct `(function_name, output_type, input_types, constants, record_count)` per pipeline branch, sourced from `version_keys` in `_record_metadata` (does not require scilineage tracking).

---

### 4. No function or pipeline registry

There is no `Pipeline` class, no `PipelineStep`, and no mechanism to register or discover analysis functions. A pipeline builder GUI would need:

- A way to register functions with metadata (name, description, input parameter names and types, output types)
- A serializable `PipelineStep` class: function reference + input spec + output spec
- A `Pipeline` class: an ordered sequence of steps, serializable to JSON/YAML
- A way to reconstruct and replay a pipeline definition without executing Python code

---

### 5. No progress/event callbacks in `for_each()`

`for_each()` currently only prints to stdout. A GUI needs structured progress events to display a progress bar or react to intermediate results.

**Needed:** A callback or event hook API — e.g., `on_start`, `on_iteration(n, total, metadata)`, `on_save(variable_type, metadata)`, `on_error(metadata, exc)`.

---

### 6. No multi-hop DB-side provenance traversal

`get_provenance()` gives only one hop: what function produced a given record. `scilineage.get_upstream_lineage()` traverses the full chain but only works for in-memory `LineageFcnResult` objects.

For saved DB records, there is no recursive "full upstream tree" query against the stored `_lineage` table. A lineage graph visualization requires following edges through multiple saved records.

**Needed:** `db.get_upstream_provenance(record_id)` — recursive traversal of `_lineage` using `output_record_id` → `inputs[].record_id` links, returning a DAG of provenance records.

---

## On Branch Naming

An important design question: do pipeline branches need explicit user-facing names (like git branch names), or are they inherently identified by their parameter values?

**Branch naming is probably not necessary.** In git, names are needed because commit content doesn't carry semantic information about *why* branches diverge. In SciStack, the parameters themselves carry that meaning — `bandpass_filter(low_hz=20, high_hz=450)` is already self-documenting. A GUI can auto-label branches from their distinguishing constants without requiring user-supplied names.

Explicit branch names could be offered as an optional UX enhancement (let the user tag a branch as `"conservative_filter"` for easier navigation), but they are not architecturally necessary. The branch identity is the unique combination of `(function_name, constants)`.

---

## Summary Table

| Capability | Status | Notes |
|---|---|---|
| Data browsing (load, filter, peek) | **Ready** | `load()`, `load_all()`, `head()`, `where=` |
| Schema/data discovery | **Ready** | `distinct_schema_values/combinations()` |
| One-hop provenance | **Ready** | `get_provenance()`, `get_provenance_by_schema()` |
| Pipeline skeleton graph | **Ready** | `get_pipeline_structure()` |
| Version history | **Ready** | `list_versions()` |
| Manual intervention hook | **Ready** | `scilineage.manual()` |
| Variable type discovery from DB | **Ready** | `db.list_variables()` |
| Parameter sweep branch isolation | **Ready** | `__constants` in version_keys + `branch_params` column |
| Branch-aware load API | **Ready** | `load(low_hz=20)`, `AmbiguousVersionError`, `list_versions(branch_params)` |
| Variant exclusion | **Ready** | `db.exclude_variant()`, `db.include_variant()` |
| Pipeline branch enumeration | **Ready** | `db.list_pipeline_variants()` |
| Function/pipeline registry | **Missing** | No `Pipeline`, `PipelineStep`, or function registry |
| Progress/event callbacks | **Missing** | `for_each()` only prints to stdout |
| Multi-hop DB provenance traversal | **Missing** | Need `db.get_upstream_provenance(record_id)` |
| Explicit branch naming | **Optional** | Parameters are self-documenting; names are a UX enhancement |
