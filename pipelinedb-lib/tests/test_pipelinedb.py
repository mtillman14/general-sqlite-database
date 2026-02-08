"""Tests for PipelineDB."""

import os
import tempfile
from pathlib import Path

import pytest

from pipelinedb import PipelineDB


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_pipeline.db"
        db = PipelineDB(db_path)
        yield db
        db.close()


class TestPipelineDBInit:
    """Tests for PipelineDB initialization."""

    def test_creates_database_file(self):
        """Database file should be created on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "new.db"
            assert not db_path.exists()

            db = PipelineDB(db_path)
            assert db_path.exists()
            db.close()

    def test_creates_lineage_table(self, temp_db):
        """Lineage table should be created on init."""
        cursor = temp_db._conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='lineage'"
        )
        assert cursor.fetchone() is not None

    def test_creates_indexes(self, temp_db):
        """Indexes should be created on init."""
        cursor = temp_db._conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_lineage_hash'"
        )
        assert cursor.fetchone() is not None

    def test_schema_has_new_columns(self, temp_db):
        """Lineage table should have schema_keys and output_content_hash columns."""
        cursor = temp_db._conn.cursor()
        cursor.execute("PRAGMA table_info(lineage)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "schema_keys" in columns
        assert "output_content_hash" in columns


class TestSaveLineage:
    """Tests for save_lineage method."""

    def test_save_basic_lineage(self, temp_db):
        """Should save basic lineage record."""
        temp_db.save_lineage(
            output_record_id="record_001",
            output_type="ProcessedData",
            function_name="process",
            function_hash="abc123",
            inputs=[{"name": "arg_0", "record_id": "input_001", "type": "RawData"}],
            constants=[{"name": "threshold", "value": 0.5}],
            lineage_hash="lineage_hash_001",
            user_id="user_1",
        )

        result = temp_db.get_lineage("record_001")
        assert result is not None
        assert result["output_record_id"] == "record_001"
        assert result["output_type"] == "ProcessedData"
        assert result["function_name"] == "process"
        assert result["function_hash"] == "abc123"
        assert result["lineage_hash"] == "lineage_hash_001"
        assert result["user_id"] == "user_1"
        assert len(result["inputs"]) == 1
        assert result["inputs"][0]["name"] == "arg_0"

    def test_save_without_optional_fields(self, temp_db):
        """Should save lineage without optional fields."""
        temp_db.save_lineage(
            output_record_id="record_002",
            output_type="Data",
            function_name="compute",
            function_hash="def456",
            inputs=[],
            constants=[],
        )

        result = temp_db.get_lineage("record_002")
        assert result is not None
        assert result["lineage_hash"] is None
        assert result["user_id"] is None
        assert result["schema_keys"] is None
        assert result["output_content_hash"] is None

    def test_save_with_schema_keys(self, temp_db):
        """Should save and retrieve schema keys."""
        temp_db.save_lineage(
            output_record_id="record_sk",
            output_type="ProcessedData",
            function_name="process",
            function_hash="abc123",
            inputs=[],
            constants=[],
            schema_keys={"subject": "S01", "session": "1"},
        )

        result = temp_db.get_lineage("record_sk")
        assert result["schema_keys"] == {"subject": "S01", "session": "1"}

    def test_save_with_content_hash(self, temp_db):
        """Should save and retrieve output content hash."""
        temp_db.save_lineage(
            output_record_id="record_ch",
            output_type="ProcessedData",
            function_name="process",
            function_hash="abc123",
            inputs=[],
            constants=[],
            output_content_hash="contenthash_abc",
        )

        result = temp_db.get_lineage("record_ch")
        assert result["output_content_hash"] == "contenthash_abc"

    def test_save_with_all_new_fields(self, temp_db):
        """Should save and retrieve both schema keys and content hash."""
        temp_db.save_lineage(
            output_record_id="record_full",
            output_type="ProcessedData",
            function_name="process",
            function_hash="abc123",
            inputs=[{"name": "x", "record_id": "in1", "type": "Raw", "content_hash": "ch_in1"}],
            constants=[],
            lineage_hash="lh_full",
            schema_keys={"subject": "S02", "session": "3"},
            output_content_hash="ch_out",
        )

        result = temp_db.get_lineage("record_full")
        assert result["schema_keys"] == {"subject": "S02", "session": "3"}
        assert result["output_content_hash"] == "ch_out"
        assert result["inputs"][0]["content_hash"] == "ch_in1"

    def test_upsert_on_conflict(self, temp_db):
        """Should update existing record on conflict."""
        # First save
        temp_db.save_lineage(
            output_record_id="record_003",
            output_type="TypeA",
            function_name="func_v1",
            function_hash="hash_v1",
            inputs=[],
            constants=[],
        )

        # Second save with same record_id
        temp_db.save_lineage(
            output_record_id="record_003",
            output_type="TypeB",
            function_name="func_v2",
            function_hash="hash_v2",
            inputs=[{"name": "x", "record_id": "y", "type": "Z"}],
            constants=[],
            schema_keys={"subject": "S01"},
            output_content_hash="updated_hash",
        )

        result = temp_db.get_lineage("record_003")
        assert result["output_type"] == "TypeB"
        assert result["function_name"] == "func_v2"
        assert len(result["inputs"]) == 1
        assert result["schema_keys"] == {"subject": "S01"}
        assert result["output_content_hash"] == "updated_hash"


class TestFindByLineageHash:
    """Tests for find_by_lineage_hash method."""

    def test_find_existing_hash(self, temp_db):
        """Should find records by lineage hash."""
        temp_db.save_lineage(
            output_record_id="record_a",
            output_type="TypeA",
            function_name="func",
            function_hash="fhash",
            inputs=[],
            constants=[],
            lineage_hash="shared_hash",
        )

        results = temp_db.find_by_lineage_hash("shared_hash")
        assert results is not None
        assert len(results) == 1
        assert results[0]["output_record_id"] == "record_a"

    def test_find_multiple_with_same_hash(self, temp_db):
        """Should find multiple records with same lineage hash."""
        for i in range(3):
            temp_db.save_lineage(
                output_record_id=f"record_{i}",
                output_type="Type",
                function_name="func",
                function_hash="fhash",
                inputs=[],
                constants=[],
                lineage_hash="common_hash",
            )

        results = temp_db.find_by_lineage_hash("common_hash")
        assert results is not None
        assert len(results) == 3

    def test_find_nonexistent_hash(self, temp_db):
        """Should return None for nonexistent hash."""
        results = temp_db.find_by_lineage_hash("nonexistent")
        assert results is None

    def test_find_by_lineage_hash_with_schema_filter(self, temp_db):
        """Should filter by schema keys when provided."""
        temp_db.save_lineage(
            output_record_id="r_s01",
            output_type="Type",
            function_name="func",
            function_hash="fhash",
            inputs=[],
            constants=[],
            lineage_hash="same_hash",
            schema_keys={"subject": "S01"},
        )
        temp_db.save_lineage(
            output_record_id="r_s02",
            output_type="Type",
            function_name="func",
            function_hash="fhash",
            inputs=[],
            constants=[],
            lineage_hash="same_hash",
            schema_keys={"subject": "S02"},
        )

        # Without filter: both
        all_results = temp_db.find_by_lineage_hash("same_hash")
        assert len(all_results) == 2

        # With filter: only S01
        filtered = temp_db.find_by_lineage_hash("same_hash", schema_keys={"subject": "S01"})
        assert len(filtered) == 1
        assert filtered[0]["output_record_id"] == "r_s01"


class TestFindBySchema:
    """Tests for find_by_schema method."""

    def test_find_by_single_schema_key(self, temp_db):
        """Should find records by a single schema key."""
        temp_db.save_lineage(
            output_record_id="r1",
            output_type="TypeA",
            function_name="func1",
            function_hash="h1",
            inputs=[],
            constants=[],
            schema_keys={"subject": "S01", "session": "1"},
        )
        temp_db.save_lineage(
            output_record_id="r2",
            output_type="TypeB",
            function_name="func2",
            function_hash="h2",
            inputs=[],
            constants=[],
            schema_keys={"subject": "S01", "session": "2"},
        )
        temp_db.save_lineage(
            output_record_id="r3",
            output_type="TypeA",
            function_name="func1",
            function_hash="h1",
            inputs=[],
            constants=[],
            schema_keys={"subject": "S02", "session": "1"},
        )

        results = temp_db.find_by_schema(subject="S01")
        assert len(results) == 2
        record_ids = {r["output_record_id"] for r in results}
        assert record_ids == {"r1", "r2"}

    def test_find_by_multiple_schema_keys(self, temp_db):
        """Should find records by multiple schema keys."""
        temp_db.save_lineage(
            output_record_id="r1",
            output_type="TypeA",
            function_name="func",
            function_hash="h",
            inputs=[],
            constants=[],
            schema_keys={"subject": "S01", "session": "1"},
        )
        temp_db.save_lineage(
            output_record_id="r2",
            output_type="TypeA",
            function_name="func",
            function_hash="h",
            inputs=[],
            constants=[],
            schema_keys={"subject": "S01", "session": "2"},
        )

        results = temp_db.find_by_schema(subject="S01", session="1")
        assert len(results) == 1
        assert results[0]["output_record_id"] == "r1"

    def test_find_by_schema_excludes_null_schema(self, temp_db):
        """Should not return records with no schema_keys."""
        temp_db.save_lineage(
            output_record_id="r_no_schema",
            output_type="Type",
            function_name="func",
            function_hash="h",
            inputs=[],
            constants=[],
        )
        temp_db.save_lineage(
            output_record_id="r_with_schema",
            output_type="Type",
            function_name="func",
            function_hash="h",
            inputs=[],
            constants=[],
            schema_keys={"subject": "S01"},
        )

        results = temp_db.find_by_schema(subject="S01")
        assert len(results) == 1
        assert results[0]["output_record_id"] == "r_with_schema"

    def test_find_by_schema_empty_result(self, temp_db):
        """Should return empty list when no matches."""
        results = temp_db.find_by_schema(subject="nonexistent")
        assert results == []


class TestGetPipelineStructure:
    """Tests for get_pipeline_structure method."""

    def test_single_function_structure(self, temp_db):
        """Should return structure for a single function."""
        temp_db.save_lineage(
            output_record_id="r1",
            output_type="ProcessedData",
            function_name="normalize",
            function_hash="nh",
            inputs=[{"name": "x", "type": "RawData", "source_type": "variable"}],
            constants=[],
        )

        structure = temp_db.get_pipeline_structure()
        assert len(structure) == 1
        assert structure[0]["function_name"] == "normalize"
        assert structure[0]["output_type"] == "ProcessedData"
        assert "RawData" in structure[0]["input_types"]

    def test_deduplicates_same_function(self, temp_db):
        """Should deduplicate when same function runs multiple times."""
        for i in range(3):
            temp_db.save_lineage(
                output_record_id=f"r{i}",
                output_type="ProcessedData",
                function_name="normalize",
                function_hash="nh",
                inputs=[{"name": "x", "type": "RawData", "source_type": "variable"}],
                constants=[],
                schema_keys={"subject": f"S{i:02d}"},
            )

        structure = temp_db.get_pipeline_structure()
        assert len(structure) == 1

    def test_multi_step_pipeline_structure(self, temp_db):
        """Should show multi-step pipeline structure."""
        temp_db.save_lineage(
            output_record_id="r1",
            output_type="NormalizedData",
            function_name="normalize",
            function_hash="nh",
            inputs=[{"name": "x", "type": "RawData", "source_type": "variable"}],
            constants=[],
        )
        temp_db.save_lineage(
            output_record_id="r2",
            output_type="FinalResult",
            function_name="analyze",
            function_hash="ah",
            inputs=[{"name": "x", "type": "NormalizedData", "source_type": "variable"}],
            constants=[{"name": "factor", "value_type": "int"}],
        )

        structure = temp_db.get_pipeline_structure()
        assert len(structure) == 2
        func_names = {s["function_name"] for s in structure}
        assert func_names == {"normalize", "analyze"}

    def test_different_function_hash_separate_entries(self, temp_db):
        """Different function hashes should produce separate structure entries."""
        temp_db.save_lineage(
            output_record_id="r1",
            output_type="Result",
            function_name="process",
            function_hash="v1_hash",
            inputs=[{"name": "x", "type": "Raw", "source_type": "variable"}],
            constants=[],
        )
        temp_db.save_lineage(
            output_record_id="r2",
            output_type="Result",
            function_name="process",
            function_hash="v2_hash",
            inputs=[{"name": "x", "type": "Raw", "source_type": "variable"}],
            constants=[],
        )

        structure = temp_db.get_pipeline_structure()
        assert len(structure) == 2

    def test_empty_pipeline_structure(self, temp_db):
        """Should return empty list when no lineage exists."""
        structure = temp_db.get_pipeline_structure()
        assert structure == []


class TestGetLineage:
    """Tests for get_lineage method."""

    def test_get_existing_lineage(self, temp_db):
        """Should get lineage for existing record."""
        temp_db.save_lineage(
            output_record_id="test_record",
            output_type="TestType",
            function_name="test_func",
            function_hash="test_hash",
            inputs=[{"a": 1}],
            constants=[{"b": 2}],
            lineage_hash="lhash",
            user_id="user",
        )

        result = temp_db.get_lineage("test_record")
        assert result is not None
        assert result["output_record_id"] == "test_record"
        assert result["inputs"] == [{"a": 1}]
        assert result["constants"] == [{"b": 2}]

    def test_get_nonexistent_lineage(self, temp_db):
        """Should return None for nonexistent record."""
        result = temp_db.get_lineage("nonexistent")
        assert result is None


class TestSaveEphemeral:
    """Tests for save_ephemeral method."""

    def test_save_ephemeral(self, temp_db):
        """Should save ephemeral lineage."""
        temp_db.save_ephemeral(
            ephemeral_id="ephemeral:abc123",
            variable_type="IntermediateData",
            function_name="transform",
            function_hash="thash",
            inputs=[{"name": "x", "record_id": "input_1", "type": "Raw"}],
            constants=[],
            user_id="user",
        )

        result = temp_db.get_lineage("ephemeral:abc123")
        assert result is not None
        assert result["output_type"] == "IntermediateData"

    def test_save_ephemeral_with_schema_keys(self, temp_db):
        """Should save ephemeral lineage with schema keys."""
        temp_db.save_ephemeral(
            ephemeral_id="ephemeral:with_schema",
            variable_type="IntermediateData",
            function_name="transform",
            function_hash="thash",
            inputs=[],
            constants=[],
            schema_keys={"subject": "S01"},
        )

        result = temp_db.get_lineage("ephemeral:with_schema")
        assert result["schema_keys"] == {"subject": "S01"}

    def test_ephemeral_idempotent(self, temp_db):
        """Should not update if ephemeral already exists."""
        temp_db.save_ephemeral(
            ephemeral_id="ephemeral:xyz",
            variable_type="TypeA",
            function_name="func1",
            function_hash="hash1",
            inputs=[],
            constants=[],
        )

        # Try to save again with different data
        temp_db.save_ephemeral(
            ephemeral_id="ephemeral:xyz",
            variable_type="TypeB",
            function_name="func2",
            function_hash="hash2",
            inputs=[{"x": 1}],
            constants=[],
        )

        # Should still have original data
        result = temp_db.get_lineage("ephemeral:xyz")
        assert result["output_type"] == "TypeA"
        assert result["function_name"] == "func1"


class TestHasLineage:
    """Tests for has_lineage method."""

    def test_has_lineage_true(self, temp_db):
        """Should return True for existing record."""
        temp_db.save_lineage(
            output_record_id="exists",
            output_type="Type",
            function_name="func",
            function_hash="hash",
            inputs=[],
            constants=[],
        )

        assert temp_db.has_lineage("exists") is True

    def test_has_lineage_false(self, temp_db):
        """Should return False for nonexistent record."""
        assert temp_db.has_lineage("does_not_exist") is False


class TestSchemaUpgrade:
    """Tests for schema migration from older databases."""

    def test_opens_old_schema_database(self):
        """Should add new columns when opening a database without them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "old.db"

            # Create a database with old schema (no schema_keys/output_content_hash)
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            conn.execute("""
                CREATE TABLE lineage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    output_record_id TEXT UNIQUE NOT NULL,
                    output_type TEXT NOT NULL,
                    lineage_hash TEXT,
                    function_name TEXT NOT NULL,
                    function_hash TEXT NOT NULL,
                    inputs TEXT NOT NULL,
                    constants TEXT NOT NULL,
                    user_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lineage_hash ON lineage(lineage_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_output_type ON lineage(output_type)"
            )
            conn.commit()
            conn.close()

            # Open with PipelineDB — should add new columns
            db = PipelineDB(db_path)
            cursor = db._conn.cursor()
            cursor.execute("PRAGMA table_info(lineage)")
            columns = {row[1] for row in cursor.fetchall()}
            assert "schema_keys" in columns
            assert "output_content_hash" in columns
            db.close()


class TestContextManager:
    """Tests for context manager protocol."""

    def test_context_manager(self):
        """Should work as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "ctx.db"

            with PipelineDB(db_path) as db:
                db.save_lineage(
                    output_record_id="ctx_record",
                    output_type="Type",
                    function_name="func",
                    function_hash="hash",
                    inputs=[],
                    constants=[],
                )
                assert db.has_lineage("ctx_record")

            # Connection should be closed after exiting context
            assert db._conn is None
