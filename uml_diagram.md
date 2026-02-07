# SciDB Framework - UML Class Diagram

## Overview

SciDB is a scientific data versioning framework with two main layers:

- **Core Layer (scidb)**: Database management and variable persistence
- **Thunk Layer**: Lazy evaluation and lineage tracking

---

## Class Diagram (Mermaid)

```mermaid
classDiagram
    direction TB

    %% ===== CORE LAYER =====
    class BaseVariable {
        <<abstract>>
        +data: Any
        +schema_version: int
        -_vhash: str
        -_metadata: dict
        -_content_hash: str
        -_lineage_hash: str
        +to_db()* DataFrame
        +from_db(df)* Any
        +save(db, **metadata) str
        +load(db, version, **metadata) BaseVariable
        +table_name() str
        +get_preview() str
    }

    class DatabaseManager {
        +db_path: Path
        +lineage_mode: str
        +connection: Connection
        -_registered_types: dict
        +register(variable_class)
        +save(variable, metadata, lineage) str
        +load(variable_class, metadata, version) BaseVariable
        +list_versions(**metadata) list
        +get_provenance(version, **metadata) dict
        +get_full_lineage(version, max_depth) dict
        +cache_computation(cache_key, ...)
        +get_cached_computation(cache_key) BaseVariable
        +invalidate_cache(function_name) int
    }

    %% ===== THUNK LAYER =====
    class Thunk {
        +fcn: Callable
        +unpack_outputs: bool
        +unwrap: bool
        +hash: str
        +pipeline_thunks: tuple
        +__call__(*args, **kwargs) ThunkOutput
    }

    class PipelineThunk {
        +thunk: Thunk
        +inputs: dict
        +outputs: tuple
        +unwrap: bool
        +hash: str
        +is_complete: bool
        +compute_cache_key() str
    }

    class ThunkOutput {
        +pipeline_thunk: PipelineThunk
        +output_num: int
        +is_complete: bool
        +data: Any
        +hash: str
        +was_cached: bool
        +cached_id: str
    }

    class LineageRecord {
        +function_name: str
        +function_hash: str
        +inputs: list
        +constants: list
        +to_dict() dict
        +from_dict(data)$ LineageRecord
    }

    class CacheBackend {
        <<interface>>
        +get_cached(cache_key) list
    }

    %% ===== EXCEPTIONS =====
    class SciDBError {
        <<exception>>
    }
    class NotRegisteredError {
        <<exception>>
    }
    class NotFoundError {
        <<exception>>
    }
    class DatabaseNotConfiguredError {
        <<exception>>
    }
    class ReservedMetadataKeyError {
        <<exception>>
    }
    class UnsavedIntermediateError {
        <<exception>>
    }

    %% ===== RELATIONSHIPS =====

    %% Exception hierarchy
    SciDBError <|-- NotRegisteredError
    SciDBError <|-- NotFoundError
    SciDBError <|-- DatabaseNotConfiguredError
    SciDBError <|-- ReservedMetadataKeyError
    SciDBError <|-- UnsavedIntermediateError

    %% Core relationships
    DatabaseManager "1" --> "*" BaseVariable : manages
    DatabaseManager "1" --> "*" LineageRecord : stores
    DatabaseManager ..|> CacheBackend : implements

    %% Thunk system
    Thunk "1" *-- "*" PipelineThunk : creates
    PipelineThunk "1" *-- "1..*" ThunkOutput : produces
    ThunkOutput "*" --> "1" PipelineThunk : references

    %% Cross-layer
    BaseVariable "1" o-- "0..1" ThunkOutput : wraps
    BaseVariable ..> LineageRecord : extracts
    BaseVariable ..> DatabaseManager : uses
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER APPLICATION                              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌─────────────────────────────┐     ┌─────────────────────────────────┐
│     BaseVariable            │     │          @thunk decorator        │
│  (User-defined subclasses)  │     │    (Function wrapper)            │
│                             │     │                                  │
│  • ScalarValue              │     │  Thunk ──creates──► PipelineThunk│
│  • ArrayValue               │     │                          │       │
│  • MatrixValue              │     │                     produces     │
│  • (custom types...)        │     │                          ▼       │
└─────────────────────────────┘     │                    ThunkOutput   │
              │                     └─────────────────────────────────┘
              │ wraps                            │
              ◄──────────────────────────────────┘
              │
              │ save()/load()
              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       DatabaseManager                                │
│                                                                      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────────┐ │
│  │_registered_  │ │  _data       │ │  _lineage    │ │_computation_│ │
│  │   types      │ │(content-addr)│ │              │ │   cache     │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ └─────────────┘ │
│                                                                      │
│                        SQLite Database                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Data Flows

### 1. Save Operation

```
BaseVariable.save()
    │
    ├─► If data is ThunkOutput: extract LineageRecord
    │
    ├─► to_db() → DataFrame
    │
    └─► DatabaseManager.save()
            │
            ├─► serialize DataFrame (Parquet)
            ├─► compute content_hash
            ├─► generate vhash
            ├─► store in _data table (deduplicated)
            ├─► store metadata in variable table
            ├─► store LineageRecord in _lineage table
            └─► populate _computation_cache
```

### 2. Lineage Tracking Flow

```
@thunk
def process(data):          ─────► Thunk (wraps function)
    return transformed                  │
                                        │ __call__()
                                        ▼
process(my_var)             ─────► PipelineThunk (captures inputs)
                                        │
                                        │ execute
                                        ▼
result = ...                ─────► ThunkOutput (lazy result + lineage)
                                        │
                                        │ wrap
                                        ▼
MyVariable(data=result)     ─────► BaseVariable with lineage
                                        │
                                        │ save()
                                        ▼
                            ─────► LineageRecord stored in DB
```

---

## Relationship Legend

| Symbol  | Meaning                        |
| ------- | ------------------------------ |
| `<\|--` | Inheritance                    |
| `*--`   | Composition (lifecycle owned)  |
| `o--`   | Aggregation (shared lifecycle) |
| `-->`   | Association                    |
| `..>`   | Dependency                     |
| `..\|>` | Implements interface           |

---

## Files

- **PlantUML**: `uml_diagram.puml` - Open with PlantUML extension or [plantuml.com](https://www.plantuml.com/plantuml)
- **This file**: `uml_diagram.md` - View in any Markdown renderer with Mermaid support
