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
        )

        result = temp_db.get_lineage("record_003")
        assert result["output_type"] == "TypeB"
        assert result["function_name"] == "func_v2"
        assert len(result["inputs"]) == 1


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
