"""
Shared fixtures for scistack-gui tests.

Sets up a real DuckDB database populated with variable data and a for_each
run so that the full API surface can be exercised end-to-end.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scifor as _scifor

# Make sure the local package is importable from an editable install.
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from scidb import BaseVariable, configure_database, for_each
from scidb.database import _local

import scistack_gui.db as _gui_db
from scistack_gui import registry as _registry
from scistack_gui.app import create_app


# ---------------------------------------------------------------------------
# Test variable classes — defined at module level so they are always present
# in BaseVariable._all_subclasses when the test client is created.
# ---------------------------------------------------------------------------

class RawSignal(BaseVariable):
    pass


class FilteredSignal(BaseVariable):
    pass


# ---------------------------------------------------------------------------
# Pipeline function used to populate test DB
# ---------------------------------------------------------------------------

def bandpass_filter(signal, low_hz):
    """Simple filter stub: scales signal by low_hz constant."""
    return np.asarray(signal, dtype=float) * float(low_hz)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_db_state():
    """
    Reset all module-level singletons between tests so no state leaks.
    Runs before (via yield) and after each test.
    """
    # Pre-test: clear everything
    if hasattr(_local, "database"):
        delattr(_local, "database")
    _gui_db._db = None
    _gui_db._db_path = None
    _scifor.set_schema([])
    # Keep only the test functions registered across tests
    _registry._functions.clear()

    yield

    # Post-test: clean up again
    if hasattr(_local, "database"):
        delattr(_local, "database")
    _gui_db._db = None
    _gui_db._db_path = None
    _scifor.set_schema([])
    _registry._functions.clear()


@pytest.fixture
def tmp_db_path(tmp_path):
    """Provide a temporary .duckdb path (file is NOT created yet)."""
    return tmp_path / "test.duckdb"


@pytest.fixture
def populated_db(tmp_path):
    """
    Real DuckDB with subject/session schema, 4 RawSignal records,
    and a for_each run producing FilteredSignal (with constant low_hz=20).

    Also sets scistack_gui.db._db / _db_path so API endpoints work.
    """
    db_path = tmp_path / "test.duckdb"
    db = configure_database(db_path, ["subject", "session"])

    # Seed raw data
    for subj in [1, 2]:
        for sess in ["pre", "post"]:
            RawSignal.save(np.random.randn(10), subject=subj, session=sess)

    # Run pipeline so list_pipeline_variants() returns results
    for_each(
        bandpass_filter,
        inputs={"signal": RawSignal, "low_hz": 20},
        outputs=[FilteredSignal],
        subject=[1, 2],
        session=["pre", "post"],
    )

    # Point the GUI db module at this database
    _gui_db._db = db
    _gui_db._db_path = db_path

    yield db

    db.close()


@pytest.fixture
def layout_path(tmp_path, populated_db):
    """
    Return the layout file path that layout.py will read/write.
    (It derives the path from get_db_path(), which is already set by populated_db.)
    """
    return _gui_db.get_db_path().with_suffix(".layout.json")


@pytest.fixture
def client(populated_db):
    """FastAPI TestClient backed by the populated database."""
    # Register the test function so /api/registry and /api/run can see it
    _registry._functions["bandpass_filter"] = bandpass_filter
    app = create_app()
    with TestClient(app) as c:
        yield c
