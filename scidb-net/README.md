# scidb-net

Network client-server layer for SciDB. Wraps `DatabaseManager` behind a FastAPI server and provides a drop-in HTTP client so existing `BaseVariable.save()`/`.load()` and thunk caching work transparently over the network.

## Installation

```bash
pip install scidb-net
```

## Server

```bash
# Via environment variables
export SCIDB_DATASET_DB_PATH=/data/experiment.duckdb
export SCIDB_DATASET_SCHEMA_KEYS='["subject", "session"]'
export SCIDB_PIPELINE_DB_PATH=/data/pipeline.db
scidb-server
```

```python
# Programmatic
from scidbnet import create_app
import uvicorn

app = create_app("/data/experiment.duckdb", ["subject", "session"], "/data/pipeline.db")
uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Client

```python
from scidbnet import configure_remote_database

configure_remote_database("http://localhost:8000")

# All existing code works unchanged
RawSignal.save(data, subject=1, session="A")
loaded = RawSignal.load(subject=1, session="A")
```

## Wire Format

- **DataFrames / numpy arrays**: Arrow IPC
- **Scalars / dicts / lists**: JSON
- **Envelope**: `4-byte header length | JSON header | body bytes`

## API

All endpoints live under `/api/v1/`. See `scidbnet/_types.py` for request/response models.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check |
| POST | `/register` | Register a variable type |
| POST | `/save` | Save data + metadata + lineage |
| POST | `/load` | Load a single variable |
| POST | `/load_all` | Load all matching variables |
| POST | `/find_by_lineage` | Cache lookup by lineage hash |
