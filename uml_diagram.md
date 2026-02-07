# SciDB Framework - UML Class Diagram

## Overview

SciDB is a scientific data versioning framework with two main layers:

- **Core Layer (scidb)**: Database management and variable persistence
- **Thunk Layer**: Lineage tracking and computation caching

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
        -_record_id: str
        -_metadata: dict
        -_content_hash: str
        -_lineage_hash: str
        +to_db()* DataFrame
        +from_db(df)* Any
        +save(data, index, **metadata)$ str
        +load(version, loc, iloc, **metadata)$ BaseVariable
        +table_name()$ str
    }

    class DatabaseManager {
        +dataset_db_path: Path
        +lineage_mode: str
        -_duck: SciDuck
        -_pipeline_db: PipelineDB
        -_registered_types: dict
        +register(variable_class)
        +save(variable, metadata, lineage) str
        +save_variable(variable_class, data, **metadata) str
        +load(variable_class, metadata, version) BaseVariable
        +list_versions(**metadata) list
        +get_provenance(version, **metadata) dict
        +find_by_lineage(pipeline_thunk) list
        +has_lineage(record_id) bool
    }

    %% ===== THUNK LAYER =====
    class Thunk {
        +fcn: Callable
        +unpack_output: bool
        +unwrap: bool
        +hash: str
        +pipeline_thunks: tuple
        +query: Any$
        +__call__(*args, **kwargs) ThunkOutput
    }

    class PipelineThunk {
        +thunk: Thunk
        +inputs: dict
        +outputs: tuple
        +unwrap: bool
        +hash: str
        +is_complete: bool
        +compute_lineage_hash() str
    }

    class ThunkOutput {
        +pipeline_thunk: PipelineThunk
        +output_num: int
        +is_complete: bool
        +data: Any
        +hash: str
    }

    class LineageRecord {
        +function_name: str
        +function_hash: str
        +inputs: list
        +constants: list
    }

    %% ===== STORAGE BACKENDS =====
    class SciDuck {
        +db_path: str
        +dataset_schema: list
        +save(name, data, **schema_keys)
        +load(name, **schema_keys)
    }

    class PipelineDB {
        +db_path: Path
        +save_lineage(...)
        +find_by_lineage_hash(hash) list
        +get_lineage(record_id) dict
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
    DatabaseManager "1" *-- "1" SciDuck : data storage
    DatabaseManager "1" *-- "1" PipelineDB : lineage storage

    %% Thunk system
    Thunk "1" *-- "*" PipelineThunk : creates
    PipelineThunk "1" *-- "1..*" ThunkOutput : produces
    ThunkOutput "*" --> "1" PipelineThunk : references

    %% Cross-layer
    BaseVariable "1" o-- "0..1" ThunkOutput : wraps
    BaseVariable ..> LineageRecord : extracts
    BaseVariable ..> DatabaseManager : uses
    Thunk ..> DatabaseManager : queries (Thunk.query)
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
│  ┌──────────────────────────┐    ┌──────────────────────────┐       │
│  │   SciDuck (DuckDB)       │    │   PipelineDB (SQLite)    │       │
│  │                          │    │                          │       │
│  │  - Data storage          │◄───│  - Lineage records       │       │
│  │  - Type registration     │ref │  - Cache lookups         │       │
│  │  - Version metadata      │    │  - Ephemeral entries     │       │
│  └──────────────────────────┘    └──────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Data Flows

### 1. Save Operation

```
BaseVariable.save(data, **metadata)
    │
    ├─► If data is ThunkOutput: extract LineageRecord
    │
    ├─► Native storage: SciDuck infers DuckDB types
    │   OR Custom: to_db() → DataFrame
    │
    └─► DatabaseManager.save()
            │
            ├─► compute content_hash
            ├─► generate record_id
            ├─► store data in SciDuck (DuckDB)
            ├─► store LineageRecord in PipelineDB (SQLite)
            └─► return record_id
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
result = ...                ─────► ThunkOutput (result + lineage)
                                        │
                                        │ save()
                                        ▼
                            ─────► LineageRecord stored in PipelineDB
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

---

## Files

- **PlantUML**: `uml_diagram.puml` - Open with PlantUML extension or [plantuml.com](https://www.plantuml.com/plantuml)
- **This file**: `uml_diagram.md` - View in any Markdown renderer with Mermaid support
