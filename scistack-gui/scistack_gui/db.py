"""
Shared database connection for the GUI backend.

The DatabaseManager instance is created once at startup (in __main__.py)
and shared by all API endpoints and the Jupyter kernel.
"""

from pathlib import Path
import duckdb
import scidb
from scidb.database import DatabaseManager

_db: DatabaseManager | None = None
_db_path: Path | None = None


def read_schema_keys(db_path: Path) -> list[str]:
    """
    Read the schema keys from an existing SciStack database without needing
    to know them in advance. The schema keys are stored as columns in the
    _schema table (all columns except schema_id and schema_level).
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = '_schema' "
            "AND column_name NOT IN ('schema_id', 'schema_level') "
            "ORDER BY ordinal_position"
        ).fetchall()
        return [row[0] for row in rows]
    finally:
        con.close()


def init_db(db_path: Path) -> DatabaseManager:
    """
    Open an existing SciStack database. Called once at startup.
    Reads schema keys from the DB itself so the user doesn't need to supply them.
    """
    global _db, _db_path
    schema_keys = read_schema_keys(db_path)
    _db = scidb.configure_database(db_path, schema_keys)
    _db_path = db_path

    # Migrate manual_nodes / manual_edges from JSON into DuckDB (one-time, idempotent).
    from scistack_gui import pipeline_store
    layout_path = db_path.with_suffix(".layout.json")
    pipeline_store.migrate_from_json(_db, layout_path)

    return _db


def get_db_path() -> Path:
    """Returns the path to the open database file."""
    if _db_path is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    return _db_path


def get_db() -> DatabaseManager:
    """FastAPI dependency: returns the shared db instance."""
    if _db is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    return _db
