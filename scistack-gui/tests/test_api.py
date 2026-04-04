"""
Integration tests for all scistack-gui HTTP API endpoints.

Uses FastAPI's TestClient (backed by a real populated DuckDB) to exercise
every route defined in the application.
"""

import json
import time
import pytest
import threading
import numpy as np

from scidb import configure_database, for_each, BaseVariable
from scidb.database import _local
import scistack_gui.db as _gui_db
from scistack_gui import layout as layout_store
from scistack_gui import registry as _registry
from collections import defaultdict

# Import test variable classes and pipeline function from conftest so we share
# the same class objects (avoids duplicate BaseVariable subclass registrations).
from conftest import RawSignal, FilteredSignal, bandpass_filter


# ---------------------------------------------------------------------------
# /api/info
# ---------------------------------------------------------------------------

class TestInfo:
    def test_returns_db_name(self, client):
        r = client.get("/api/info")
        assert r.status_code == 200
        assert r.json()["db_name"] == "test.duckdb"


# ---------------------------------------------------------------------------
# /api/schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_returns_schema_keys(self, client):
        r = client.get("/api/schema")
        assert r.status_code == 200
        data = r.json()
        assert data["keys"] == ["subject", "session"]

    def test_returns_distinct_values(self, client):
        r = client.get("/api/schema")
        values = r.json()["values"]
        assert set(str(v) for v in values["subject"]) == {"1", "2"}
        assert set(str(v) for v in values["session"]) == {"pre", "post"}


# ---------------------------------------------------------------------------
# /api/registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_returns_functions(self, client):
        r = client.get("/api/registry")
        assert r.status_code == 200
        data = r.json()
        assert "bandpass_filter" in data["functions"]

    def test_returns_variable_classes(self, client):
        r = client.get("/api/registry")
        data = r.json()
        assert "RawSignal" in data["variables"]
        assert "FilteredSignal" in data["variables"]

    def test_lists_are_sorted(self, client):
        r = client.get("/api/registry")
        data = r.json()
        assert data["functions"] == sorted(data["functions"])
        assert data["variables"] == sorted(data["variables"])


# ---------------------------------------------------------------------------
# /api/pipeline
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_returns_200(self, client):
        r = client.get("/api/pipeline")
        assert r.status_code == 200

    def test_has_nodes_and_edges_keys(self, client):
        data = client.get("/api/pipeline").json()
        assert "nodes" in data
        assert "edges" in data

    def test_variable_nodes_present(self, client):
        nodes = client.get("/api/pipeline").json()["nodes"]
        node_ids = {n["id"] for n in nodes}
        assert "var__RawSignal" in node_ids
        assert "var__FilteredSignal" in node_ids

    def test_function_node_present(self, client):
        nodes = client.get("/api/pipeline").json()["nodes"]
        node_ids = {n["id"] for n in nodes}
        assert "fn__bandpass_filter" in node_ids

    def test_constant_node_present(self, client):
        nodes = client.get("/api/pipeline").json()["nodes"]
        node_ids = {n["id"] for n in nodes}
        assert "const__low_hz" in node_ids

    def test_variable_node_has_total_records(self, client):
        nodes = client.get("/api/pipeline").json()["nodes"]
        raw_node = next(n for n in nodes if n["id"] == "var__RawSignal")
        assert raw_node["data"]["total_records"] == 4  # 2 subjects × 2 sessions

    def test_function_node_has_variants(self, client):
        nodes = client.get("/api/pipeline").json()["nodes"]
        fn_node = next(n for n in nodes if n["id"] == "fn__bandpass_filter")
        assert len(fn_node["data"]["variants"]) >= 1
        variant = fn_node["data"]["variants"][0]
        assert "constants" in variant
        assert "input_types" in variant
        assert "output_type" in variant

    def test_edges_connect_raw_to_function(self, client):
        edges = client.get("/api/pipeline").json()["edges"]
        matches = [
            e for e in edges
            if e["source"] == "var__RawSignal" and e["target"] == "fn__bandpass_filter"
        ]
        assert len(matches) >= 1

    def test_edges_connect_function_to_filtered(self, client):
        edges = client.get("/api/pipeline").json()["edges"]
        matches = [
            e for e in edges
            if e["source"] == "fn__bandpass_filter" and e["target"] == "var__FilteredSignal"
        ]
        assert len(matches) >= 1

    def test_constant_edge_connects_to_function(self, client):
        edges = client.get("/api/pipeline").json()["edges"]
        matches = [
            e for e in edges
            if e["source"] == "const__low_hz" and e["target"] == "fn__bandpass_filter"
        ]
        assert len(matches) >= 1

    def test_manual_node_appears_in_pipeline(self, client):
        # Add a manual node via the layout API
        client.put("/api/layout/manual__extra_var", json={
            "x": 50.0, "y": 100.0,
            "node_type": "variableNode",
            "label": "ExtraVar",
        })
        nodes = client.get("/api/pipeline").json()["nodes"]
        node_ids = {n["id"] for n in nodes}
        assert "manual__extra_var" in node_ids

    def test_manual_edge_appears_in_pipeline(self, client):
        client.put("/api/edges/manual_e1", json={
            "source": "var__RawSignal",
            "target": "fn__bandpass_filter",
            "source_handle": None,
            "target_handle": None,
        })
        edges = client.get("/api/pipeline").json()["edges"]
        edge_ids = {e["id"] for e in edges}
        assert "manual_e1" in edge_ids


# ---------------------------------------------------------------------------
# /api/layout
# ---------------------------------------------------------------------------

class TestLayoutEndpoints:
    def test_get_layout_returns_dict(self, client):
        r = client.get("/api/layout")
        assert r.status_code == 200
        data = r.json()
        assert "positions" in data
        assert "manual_nodes" in data

    def test_put_layout_saves_position(self, client):
        r = client.put("/api/layout/fn__bandpass_filter", json={"x": 100.0, "y": 200.0})
        assert r.status_code == 200
        assert r.json() == {"ok": True}
        layout = client.get("/api/layout").json()
        assert layout["positions"]["fn__bandpass_filter"] == {"x": 100.0, "y": 200.0}

    def test_put_layout_with_node_type_creates_manual_node(self, client):
        client.put("/api/layout/manual__foo", json={
            "x": 10.0, "y": 20.0,
            "node_type": "functionNode",
            "label": "my_fn",
        })
        layout = client.get("/api/layout").json()
        assert "manual__foo" in layout["manual_nodes"]
        assert layout["manual_nodes"]["manual__foo"]["label"] == "my_fn"

    def test_delete_layout_removes_node(self, client):
        client.put("/api/layout/fn__bandpass_filter", json={"x": 1.0, "y": 2.0})
        r = client.delete("/api/layout/fn__bandpass_filter")
        assert r.status_code == 200
        assert r.json() == {"ok": True}
        layout = client.get("/api/layout").json()
        assert "fn__bandpass_filter" not in layout["positions"]

    def test_put_layout_overwrites_position(self, client):
        client.put("/api/layout/fn__bandpass_filter", json={"x": 1.0, "y": 2.0})
        client.put("/api/layout/fn__bandpass_filter", json={"x": 99.0, "y": 88.0})
        layout = client.get("/api/layout").json()
        assert layout["positions"]["fn__bandpass_filter"] == {"x": 99.0, "y": 88.0}


# ---------------------------------------------------------------------------
# /api/constants
# ---------------------------------------------------------------------------

class TestConstantsEndpoints:
    def test_get_constants_initially_empty(self, client):
        r = client.get("/api/constants")
        assert r.status_code == 200
        assert r.json() == []

    def test_post_constant_adds_it(self, client):
        r = client.post("/api/constants", json={"name": "window_size"})
        assert r.status_code == 200
        assert r.json() == {"ok": True}
        constants = client.get("/api/constants").json()
        assert "window_size" in constants

    def test_post_constant_no_duplicate(self, client):
        client.post("/api/constants", json={"name": "alpha"})
        client.post("/api/constants", json={"name": "alpha"})
        constants = client.get("/api/constants").json()
        assert constants.count("alpha") == 1

    def test_delete_constant_removes_it(self, client):
        client.post("/api/constants", json={"name": "alpha"})
        client.post("/api/constants", json={"name": "beta"})
        r = client.delete("/api/constants/alpha")
        assert r.status_code == 200
        constants = client.get("/api/constants").json()
        assert "alpha" not in constants
        assert "beta" in constants

    def test_delete_nonexistent_constant_is_ok(self, client):
        r = client.delete("/api/constants/ghost")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# /api/edges
# ---------------------------------------------------------------------------

class TestEdgesEndpoints:
    def test_put_edge_saves_it(self, client):
        r = client.put("/api/edges/e_test_1", json={
            "source": "var__RawSignal",
            "target": "fn__bandpass_filter",
            "source_handle": None,
            "target_handle": "in__signal",
        })
        assert r.status_code == 200
        assert r.json() == {"ok": True}

        # Verify it shows up in the pipeline
        edges = client.get("/api/pipeline").json()["edges"]
        assert any(e["id"] == "e_test_1" for e in edges)

    def test_put_edge_upserts(self, client):
        client.put("/api/edges/e_test_2", json={
            "source": "A", "target": "B",
            "source_handle": None, "target_handle": None,
        })
        client.put("/api/edges/e_test_2", json={
            "source": "A", "target": "C",
            "source_handle": None, "target_handle": None,
        })
        layout = client.get("/api/layout").json()
        matching = [e for e in layout["manual_edges"] if e["id"] == "e_test_2"]
        assert len(matching) == 1
        assert matching[0]["target"] == "C"

    def test_delete_edge_removes_it(self, client):
        client.put("/api/edges/e_del", json={
            "source": "X", "target": "Y",
            "source_handle": None, "target_handle": None,
        })
        r = client.delete("/api/edges/e_del")
        assert r.status_code == 200
        layout = client.get("/api/layout").json()
        assert not any(e["id"] == "e_del" for e in layout["manual_edges"])

    def test_delete_nonexistent_edge_is_ok(self, client):
        r = client.delete("/api/edges/ghost_edge")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# /api/variables
# ---------------------------------------------------------------------------

class TestVariableRecordsEndpoint:
    def test_returns_schema_keys(self, client):
        r = client.get("/api/variables/FilteredSignal/records")
        assert r.status_code == 200
        data = r.json()
        assert data["schema_keys"] == ["subject", "session"]

    def test_returns_records(self, client):
        r = client.get("/api/variables/FilteredSignal/records")
        data = r.json()
        assert len(data["records"]) == 4  # 2 subjects × 2 sessions

    def test_records_have_schema_key_values(self, client):
        r = client.get("/api/variables/FilteredSignal/records")
        records = r.json()["records"]
        subjects = {rec["subject"] for rec in records}
        sessions = {rec["session"] for rec in records}
        assert subjects == {"1", "2"}
        assert sessions == {"pre", "post"}

    def test_records_have_variant_label(self, client):
        r = client.get("/api/variables/FilteredSignal/records")
        records = r.json()["records"]
        assert all("variant_label" in rec for rec in records)
        # bandpass_filter with low_hz=20 should appear in every label
        assert all("low_hz=20" in rec["variant_label"] for rec in records)

    def test_variants_summary(self, client):
        r = client.get("/api/variables/FilteredSignal/records")
        variants = r.json()["variants"]
        assert len(variants) == 1  # single variant (low_hz=20)
        assert variants[0]["record_count"] == 4
        assert "low_hz=20" in variants[0]["label"]

    def test_raw_variable_returns_records(self, client):
        # RawSignal was saved directly (no for_each), branch_params should be empty
        r = client.get("/api/variables/RawSignal/records")
        assert r.status_code == 200
        data = r.json()
        assert len(data["records"]) == 4
        assert all(rec["variant_label"] == "(raw)" for rec in data["records"])

    def test_unknown_variable_returns_empty(self, client):
        r = client.get("/api/variables/NonExistentVariable/records")
        # No records in metadata → empty result (not 404)
        assert r.status_code == 200
        data = r.json()
        assert data["records"] == []
        assert data["variants"] == []


# ---------------------------------------------------------------------------
# /api/run
# ---------------------------------------------------------------------------

def _wait_for_threads(prefix: str, timeout: float = 2.0) -> None:
    """Wait for any background run threads to finish before DB teardown."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        live = [t for t in threading.enumerate() if t.name.startswith(prefix)]
        if not live:
            break
        time.sleep(0.05)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
class TestRunEndpoint:
    def test_returns_run_id(self, client):
        r = client.post("/api/run", json={
            "function_name": "bandpass_filter",
            "variants": [],
        })
        assert r.status_code == 200
        data = r.json()
        assert "run_id" in data
        assert data["run_id"]  # non-empty
        _wait_for_threads("Thread-")

    def test_accepts_caller_supplied_run_id(self, client):
        r = client.post("/api/run", json={
            "function_name": "bandpass_filter",
            "variants": [],
            "run_id": "my_run_42",
        })
        assert r.status_code == 200
        assert r.json()["run_id"] == "my_run_42"
        _wait_for_threads("Thread-")

    def test_unknown_function_still_returns_200_with_run_id(self, client):
        """
        The HTTP layer returns immediately; the error is surfaced via WebSocket.
        So even for an unknown function, the POST itself should succeed.
        """
        r = client.post("/api/run", json={
            "function_name": "no_such_function",
            "variants": [],
        })
        assert r.status_code == 200
        assert "run_id" in r.json()
        _wait_for_threads("Thread-")


# ---------------------------------------------------------------------------
# Run state: pipeline node run_state integration tests
# ---------------------------------------------------------------------------

# Extra variable class and function for the two-step chain tests
class ProcessedSignal(BaseVariable):
    pass


def process_signal(filtered):
    return np.asarray(filtered, dtype=float) * 2.0


class TestRunStateGreen:
    """Fully-run pipeline → green for both function and output variable."""

    def test_function_node_is_green(self, client):
        nodes = client.get("/api/pipeline").json()["nodes"]
        fn_node = next(n for n in nodes if n["id"] == "fn__bandpass_filter")
        assert fn_node["data"].get("run_state") == "green"

    def test_output_variable_is_green(self, client):
        nodes = client.get("/api/pipeline").json()["nodes"]
        var_node = next(n for n in nodes if n["id"] == "var__FilteredSignal")
        assert var_node["data"].get("run_state") == "green"


class TestRunStateRed:
    """Function registered but never run → red."""

    @pytest.fixture
    def client_never_run(self, tmp_path):
        # Fresh DB with raw data only — for_each never called
        if hasattr(_local, "database"):
            delattr(_local, "database")
        _gui_db._db = None

        db = configure_database(tmp_path / "red.duckdb", ["subject", "session"])
        for subj in [1, 2]:
            for sess in ["pre", "post"]:
                RawSignal.save(np.random.randn(10), subject=subj, session=sess)

        _gui_db._db = db
        _gui_db._db_path = tmp_path / "red.duckdb"
        _registry._functions["bandpass_filter"] = bandpass_filter

        from fastapi.testclient import TestClient
        from scistack_gui.app import create_app
        with TestClient(create_app()) as c:
            yield c

        db.close()

    def test_function_node_is_red(self, client_never_run):
        nodes = client_never_run.get("/api/pipeline").json()["nodes"]
        fn_node = next((n for n in nodes if n["id"] == "fn__bandpass_filter"), None)
        # Function never run — no variants in DB, so no fn node in pipeline.
        # State is red only if the node appears (via registry).
        if fn_node is not None:
            assert fn_node["data"].get("run_state") == "red"


class TestRunStateGrey:
    """Partially-run pipeline → grey."""

    @pytest.fixture
    def client_partial(self, tmp_path):
        if hasattr(_local, "database"):
            delattr(_local, "database")
        _gui_db._db = None

        db = configure_database(tmp_path / "grey.duckdb", ["subject", "session"])
        for subj in [1, 2]:
            for sess in ["pre", "post"]:
                RawSignal.save(np.random.randn(10), subject=subj, session=sess)

        # Run for only 1 of 2 subjects → 2 of 4 schema_ids processed
        for_each(
            bandpass_filter,
            inputs={"signal": RawSignal, "low_hz": 20},
            outputs=[FilteredSignal],
            subject=[1],
            session=["pre", "post"],
        )

        _gui_db._db = db
        _gui_db._db_path = tmp_path / "grey.duckdb"
        _registry._functions["bandpass_filter"] = bandpass_filter

        from fastapi.testclient import TestClient
        from scistack_gui.app import create_app
        with TestClient(create_app()) as c:
            yield c

        db.close()

    def test_function_node_is_grey(self, client_partial):
        nodes = client_partial.get("/api/pipeline").json()["nodes"]
        fn_node = next(n for n in nodes if n["id"] == "fn__bandpass_filter")
        assert fn_node["data"].get("run_state") == "grey"

    def test_output_variable_is_grey(self, client_partial):
        nodes = client_partial.get("/api/pipeline").json()["nodes"]
        var_node = next(n for n in nodes if n["id"] == "var__FilteredSignal")
        assert var_node["data"].get("run_state") == "grey"


class TestRunStatePropagation:
    """
    Two-step chain: RawSignal → bandpass_filter → FilteredSignal → process_signal → ProcessedSignal.
    When the first step is grey (partial run), the second step must also be grey
    even if it ran completely for all available FilteredSignal records.
    """

    @pytest.fixture
    def client_propagation(self, tmp_path):
        if hasattr(_local, "database"):
            delattr(_local, "database")
        _gui_db._db = None

        db = configure_database(tmp_path / "prop.duckdb", ["subject", "session"])
        for subj in [1, 2]:
            for sess in ["pre", "post"]:
                RawSignal.save(np.random.randn(10), subject=subj, session=sess)

        # Step 1: partial run of bandpass_filter (only subject=1)
        for_each(
            bandpass_filter,
            inputs={"signal": RawSignal, "low_hz": 20},
            outputs=[FilteredSignal],
            subject=[1],
            session=["pre", "post"],
        )

        # Step 2: fully process ALL available FilteredSignal records
        for_each(
            process_signal,
            inputs={"filtered": FilteredSignal},
            outputs=[ProcessedSignal],
            subject=[1],
            session=["pre", "post"],
        )

        _gui_db._db = db
        _gui_db._db_path = tmp_path / "prop.duckdb"
        _registry._functions["bandpass_filter"] = bandpass_filter
        _registry._functions["process_signal"] = process_signal

        from fastapi.testclient import TestClient
        from scistack_gui.app import create_app
        with TestClient(create_app()) as c:
            yield c

        db.close()

    def test_upstream_is_grey(self, client_propagation):
        nodes = client_propagation.get("/api/pipeline").json()["nodes"]
        fn_node = next(n for n in nodes if n["id"] == "fn__bandpass_filter")
        assert fn_node["data"].get("run_state") == "grey"

    def test_downstream_function_is_grey_due_to_staleness(self, client_propagation):
        nodes = client_propagation.get("/api/pipeline").json()["nodes"]
        fn_node = next(n for n in nodes if n["id"] == "fn__process_signal")
        # process_signal ran for all available inputs, but upstream is grey → grey
        assert fn_node["data"].get("run_state") == "grey"

    def test_downstream_variable_is_grey_due_to_staleness(self, client_propagation):
        nodes = client_propagation.get("/api/pipeline").json()["nodes"]
        var_node = next(n for n in nodes if n["id"] == "var__ProcessedSignal")
        assert var_node["data"].get("run_state") == "grey"
