# SciDB

# Components

## Database Layer: SciDuck

DuckDB-based database management. Each variable is a unique subclass of `BaseVariable`.

## Lineage Layer: Thunk

DAG construction and lineage/provenance tracking through Haskell-style Thunk objects.

## Query by Metadata Layer: SciDB

Tightly coupled to SciDuck (for DB structure), Thunk (for PipelineThunk structure), and the SciDB BaseVariable class (for the variable names) - checks the SciDuck database to see if a PipelineThunk already exists within it. Either raises an error if no database hit found, or `result` if the database was hit.

At the start of the pipeline, set `Thunk.query = QueryByMetadata`. Then at the start of `Thunk.__call__()`, run `Thunk.query.load(var_name, **metadata)` on each input variable (aka `sciduck.load(var_name, **schema_keys)`). If it raises an error indicating the value wasn't found, then `Thunk.__call__()` proceeds to call the function.

## Implementation Layer: TBD (To Be Determined)

A thin wrapper that is tightly coupled to both the `SciDuck` and `Thunk` libraries. Implements high-level functions such as `for_each()`, database checking, and pre- and post-execution hooks.
