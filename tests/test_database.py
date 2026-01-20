"""Tests for scidb.database module."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scidb.database import (
    DatabaseManager,
    configure_database,
    get_database,
    _local,
)
from scidb.exceptions import (
    DatabaseNotConfiguredError,
    NotFoundError,
    NotRegisteredError,
)


class TestConfigureDatabase:
    """Test configure_database function."""

    def test_returns_database_manager(self, temp_db_path):
        db = configure_database(temp_db_path)
        assert isinstance(db, DatabaseManager)
        db.close()

    def test_creates_database_file(self, temp_db_path):
        db = configure_database(temp_db_path)
        assert temp_db_path.exists()
        db.close()

    def test_sets_global_database(self, temp_db_path):
        db = configure_database(temp_db_path)
        assert get_database() is db
        db.close()

    def test_can_reconfigure(self, tmp_path):
        path1 = tmp_path / "db1.sqlite"
        path2 = tmp_path / "db2.sqlite"

        db1 = configure_database(path1)
        db2 = configure_database(path2)

        assert get_database() is db2
        assert db1 is not db2

        db1.close()
        db2.close()


class TestGetDatabase:
    """Test get_database function."""

    def test_raises_when_not_configured(self, clear_global_db):
        with pytest.raises(DatabaseNotConfiguredError):
            get_database()

    def test_returns_configured_database(self, configured_db):
        assert get_database() is configured_db


class TestDatabaseManagerInit:
    """Test DatabaseManager initialization."""

    def test_creates_connection(self, temp_db_path):
        db = DatabaseManager(temp_db_path)
        assert db.connection is not None
        db.close()

    def test_creates_database_file(self, temp_db_path):
        db = DatabaseManager(temp_db_path)
        assert temp_db_path.exists()
        db.close()

    def test_accepts_string_path(self, tmp_path):
        path = str(tmp_path / "test.db")
        db = DatabaseManager(path)
        assert Path(path).exists()
        db.close()

    def test_accepts_path_object(self, temp_db_path):
        db = DatabaseManager(temp_db_path)
        assert temp_db_path.exists()
        db.close()

    def test_enables_wal_mode(self, db):
        cursor = db.connection.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode.lower() == "wal"

    def test_creates_registered_types_table(self, db):
        cursor = db.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_registered_types'"
        )
        assert cursor.fetchone() is not None

    def test_creates_version_log_table(self, db):
        cursor = db.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_version_log'"
        )
        assert cursor.fetchone() is not None


class TestDatabaseManagerContextManager:
    """Test DatabaseManager context manager protocol."""

    def test_context_manager_enter(self, temp_db_path):
        with DatabaseManager(temp_db_path) as db:
            assert db.connection is not None

    def test_context_manager_closes_connection(self, temp_db_path):
        db = DatabaseManager(temp_db_path)
        with db:
            pass
        # After context, connection should be closed
        with pytest.raises(sqlite3.ProgrammingError):
            db.connection.execute("SELECT 1")


class TestRegister:
    """Test variable type registration."""

    def test_register_creates_table(self, db, scalar_class):
        db.register(scalar_class)
        cursor = db.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='scalar_value'"
        )
        assert cursor.fetchone() is not None

    def test_register_creates_index(self, db, scalar_class):
        db.register(scalar_class)
        cursor = db.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_scalar_value_record_id'"
        )
        assert cursor.fetchone() is not None

    def test_register_adds_to_registered_types(self, db, scalar_class):
        db.register(scalar_class)
        cursor = db.connection.execute(
            "SELECT type_name, table_name, schema_version FROM _registered_types WHERE type_name = ?",
            ("ScalarValue",)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "ScalarValue"
        assert row[1] == "scalar_value"
        assert row[2] == 1

    def test_register_caches_locally(self, db, scalar_class):
        db.register(scalar_class)
        assert "ScalarValue" in db._registered_types
        assert db._registered_types["ScalarValue"] is scalar_class

    def test_register_multiple_types(self, db, scalar_class, array_class, matrix_class):
        db.register(scalar_class)
        db.register(array_class)
        db.register(matrix_class)

        assert "ScalarValue" in db._registered_types
        assert "ArrayValue" in db._registered_types
        assert "MatrixValue" in db._registered_types

    def test_register_idempotent(self, db, scalar_class):
        db.register(scalar_class)
        db.register(scalar_class)  # Should not raise

        cursor = db.connection.execute(
            "SELECT COUNT(*) FROM _registered_types WHERE type_name = ?",
            ("ScalarValue",)
        )
        assert cursor.fetchone()[0] == 1

    def test_table_has_correct_columns(self, db, scalar_class):
        db.register(scalar_class)
        cursor = db.connection.execute("PRAGMA table_info(scalar_value)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "id" in columns
        assert "record_id" in columns
        assert "schema_version" in columns
        assert "metadata" in columns
        assert "content_hash" in columns  # Data is now in _data table, referenced by content_hash
        assert "lineage_hash" in columns  # For lineage-based caching
        assert "user_id" in columns  # For cross-user attribution
        assert "created_at" in columns


class TestSave:
    """Test saving variables."""

    def test_save_returns_record_id(self, db, scalar_class):
        db.register(scalar_class)
        record_id = scalar_class.save(42, db=db, subject=1)
        assert isinstance(record_id, str)
        assert len(record_id) == 16

    def test_save_stores_data(self, db, scalar_class):
        db.register(scalar_class)
        record_id = scalar_class.save(42, db=db, subject=1)

        cursor = db.connection.execute(
            "SELECT * FROM scalar_value WHERE record_id = ?", (record_id,)
        )
        row = cursor.fetchone()
        assert row is not None

    def test_save_stores_metadata(self, db, scalar_class):
        db.register(scalar_class)
        record_id = scalar_class.save(42, db=db, subject=1, trial=2, condition="test")

        cursor = db.connection.execute(
            "SELECT metadata FROM scalar_value WHERE record_id = ?", (record_id,)
        )
        row = cursor.fetchone()
        metadata = json.loads(row[0])
        assert metadata["subject"] == 1
        assert metadata["trial"] == 2
        assert metadata["condition"] == "test"

    def test_save_stores_schema_version(self, db, scalar_class):
        db.register(scalar_class)
        record_id = scalar_class.save(42, db=db, subject=1)

        cursor = db.connection.execute(
            "SELECT schema_version FROM scalar_value WHERE record_id = ?", (record_id,)
        )
        row = cursor.fetchone()
        assert row[0] == 1

    def test_save_logs_to_version_log(self, db, scalar_class):
        db.register(scalar_class)
        record_id = scalar_class.save(42, db=db, subject=1)

        cursor = db.connection.execute(
            "SELECT type_name, table_name FROM _version_log WHERE record_id = ?",
            (record_id,)
        )
        row = cursor.fetchone()
        assert row[0] == "ScalarValue"
        assert row[1] == "scalar_value"

    def test_save_idempotent(self, db, scalar_class):
        """Saving same data+metadata twice returns same record_id."""
        db.register(scalar_class)

        record_id1 = scalar_class.save(42, db=db, subject=1)
        record_id2 = scalar_class.save(42, db=db, subject=1)

        assert record_id1 == record_id2

        # Should only have one row
        cursor = db.connection.execute(
            "SELECT COUNT(*) FROM scalar_value WHERE record_id = ?", (record_id1,)
        )
        assert cursor.fetchone()[0] == 1

    def test_save_different_data_different_record_id(self, db, scalar_class):
        db.register(scalar_class)

        record_id1 = scalar_class.save(42, db=db, subject=1)
        record_id2 = scalar_class.save(43, db=db, subject=1)

        assert record_id1 != record_id2

    def test_save_different_metadata_different_record_id(self, db, scalar_class):
        db.register(scalar_class)

        record_id1 = scalar_class.save(42, db=db, subject=1)
        record_id2 = scalar_class.save(42, db=db, subject=2)

        assert record_id1 != record_id2

    def test_save_auto_registers(self, db, scalar_class):
        """Save should auto-register the variable type."""
        # Should not raise - auto-registers on save
        record_id = scalar_class.save(42, db=db, subject=1)
        assert isinstance(record_id, str)

    def test_save_array(self, db, array_class):
        db.register(array_class)
        arr = np.array([1.0, 2.0, 3.0])
        record_id = array_class.save(arr, db=db, subject=1)
        assert isinstance(record_id, str)

    def test_save_matrix(self, db, matrix_class):
        db.register(matrix_class)
        mat = np.array([[1, 2], [3, 4]])
        record_id = matrix_class.save(mat, db=db, subject=1)
        assert isinstance(record_id, str)

    def test_save_dataframe(self, db, dataframe_class):
        db.register(dataframe_class)
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        record_id = dataframe_class.save(df, db=db, subject=1)
        assert isinstance(record_id, str)


class TestLoad:
    """Test loading variables."""

    def test_load_by_metadata(self, db, scalar_class):
        db.register(scalar_class)
        scalar_class.save(42, db=db, subject=1)

        loaded = scalar_class.load(db=db, subject=1)
        assert loaded.data == 42

    def test_load_returns_correct_type(self, db, scalar_class):
        db.register(scalar_class)
        scalar_class.save(42, db=db, subject=1)

        loaded = scalar_class.load(db=db, subject=1)
        assert isinstance(loaded, scalar_class)

    def test_load_sets_record_id(self, db, scalar_class):
        db.register(scalar_class)
        original_record_id = scalar_class.save(42, db=db, subject=1)

        loaded = scalar_class.load(db=db, subject=1)
        assert loaded.record_id == original_record_id

    def test_load_sets_metadata(self, db, scalar_class):
        db.register(scalar_class)
        scalar_class.save(42, db=db, subject=1, trial=2)

        loaded = scalar_class.load(db=db, subject=1, trial=2)
        assert loaded.metadata == {"subject": 1, "trial": 2}

    def test_load_by_record_id(self, db, scalar_class):
        db.register(scalar_class)
        record_id = scalar_class.save(42, db=db, subject=1)

        loaded = scalar_class.load(db=db, version=record_id)
        assert loaded.data == 42

    def test_load_not_found(self, db, scalar_class):
        db.register(scalar_class)
        with pytest.raises(NotFoundError):
            scalar_class.load(db=db, subject=999)

    def test_load_not_found_record_id(self, db, scalar_class):
        db.register(scalar_class)
        with pytest.raises(NotFoundError):
            scalar_class.load(db=db, version="nonexistent")

    def test_load_requires_registration(self, db, scalar_class):
        with pytest.raises(NotRegisteredError):
            scalar_class.load(db=db, subject=1)

    def test_load_multiple_matches_returns_list(self, db, scalar_class):
        db.register(scalar_class)

        # Save multiple with same partial metadata
        scalar_class.save(42, db=db, subject=1, trial=1)
        scalar_class.save(43, db=db, subject=1, trial=2)

        loaded = scalar_class.load(db=db, subject=1)
        assert isinstance(loaded, list)
        assert len(loaded) == 2

    def test_load_single_match_returns_instance(self, db, scalar_class):
        db.register(scalar_class)
        scalar_class.save(42, db=db, subject=1, trial=1)

        loaded = scalar_class.load(db=db, subject=1, trial=1)
        assert isinstance(loaded, scalar_class)
        assert not isinstance(loaded, list)

    def test_load_partial_metadata(self, db, scalar_class):
        db.register(scalar_class)

        scalar_class.save(42, db=db, subject=1, trial=1, condition="a")
        scalar_class.save(43, db=db, subject=1, trial=2, condition="b")
        scalar_class.save(44, db=db, subject=2, trial=1, condition="a")

        # Load by subject only
        loaded = scalar_class.load(db=db, subject=1)
        assert isinstance(loaded, list)
        assert len(loaded) == 2

    def test_load_array(self, db, array_class):
        db.register(array_class)
        original = np.array([1.0, 2.0, 3.0, 4.0])
        array_class.save(original, db=db, subject=1)

        loaded = array_class.load(db=db, subject=1)
        np.testing.assert_array_equal(loaded.data, original)

    def test_load_matrix(self, db, matrix_class):
        db.register(matrix_class)
        original = np.array([[1, 2, 3], [4, 5, 6]])
        matrix_class.save(original, db=db, subject=1)

        loaded = matrix_class.load(db=db, subject=1)
        np.testing.assert_array_equal(loaded.data, original)

    def test_load_dataframe(self, db, dataframe_class):
        db.register(dataframe_class)
        original = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        dataframe_class.save(original, db=db, subject=1)

        loaded = dataframe_class.load(db=db, subject=1)
        pd.testing.assert_frame_equal(loaded.data, original)


class TestLoadLatestOrdering:
    """Test that load returns latest version when multiple exist."""

    def test_load_returns_latest_first(self, db, scalar_class):
        db.register(scalar_class)

        # Save multiple versions with same metadata
        # (different data, so different record_id)
        scalar_class.save(1, db=db, subject=1, trial=1)
        scalar_class.save(2, db=db, subject=1, trial=1)
        scalar_class.save(3, db=db, subject=1, trial=1)

        loaded = scalar_class.load(db=db, subject=1, trial=1)

        # Since all have different data, should return list ordered by created_at DESC
        assert isinstance(loaded, list)
        # The last saved should be first in the list
        assert loaded[0].data == 3


class TestListVersions:
    """Test list_versions method."""

    def test_list_versions_returns_list(self, db, scalar_class):
        db.register(scalar_class)
        scalar_class.save(42, db=db, subject=1)

        versions = db.list_versions(scalar_class, subject=1)
        assert isinstance(versions, list)

    def test_list_versions_contains_record_id(self, db, scalar_class):
        db.register(scalar_class)
        record_id = scalar_class.save(42, db=db, subject=1)

        versions = db.list_versions(scalar_class, subject=1)
        assert len(versions) == 1
        assert versions[0]["record_id"] == record_id

    def test_list_versions_contains_metadata(self, db, scalar_class):
        db.register(scalar_class)
        scalar_class.save(42, db=db, subject=1, trial=2)

        versions = db.list_versions(scalar_class, subject=1)
        assert versions[0]["metadata"] == {"subject": 1, "trial": 2}

    def test_list_versions_contains_created_at(self, db, scalar_class):
        db.register(scalar_class)
        scalar_class.save(42, db=db, subject=1)

        versions = db.list_versions(scalar_class, subject=1)
        assert "created_at" in versions[0]

    def test_list_versions_multiple(self, db, scalar_class):
        db.register(scalar_class)
        scalar_class.save(1, db=db, subject=1, trial=1)
        scalar_class.save(2, db=db, subject=1, trial=2)
        scalar_class.save(3, db=db, subject=1, trial=3)

        versions = db.list_versions(scalar_class, subject=1)
        assert len(versions) == 3

    def test_list_versions_ordered_by_created_at_desc(self, db, scalar_class):
        db.register(scalar_class)
        scalar_class.save(1, db=db, subject=1, trial=1)
        scalar_class.save(2, db=db, subject=1, trial=2)
        scalar_class.save(3, db=db, subject=1, trial=3)

        versions = db.list_versions(scalar_class, subject=1)
        # Latest should be first
        assert versions[0]["metadata"]["trial"] == 3

    def test_list_versions_empty(self, db, scalar_class):
        db.register(scalar_class)
        versions = db.list_versions(scalar_class, subject=999)
        assert versions == []

    def test_list_versions_partial_metadata(self, db, scalar_class):
        db.register(scalar_class)
        scalar_class.save(1, db=db, subject=1, trial=1)
        scalar_class.save(2, db=db, subject=1, trial=2)
        scalar_class.save(3, db=db, subject=2, trial=1)

        versions = db.list_versions(scalar_class, subject=1)
        assert len(versions) == 2

    def test_list_versions_requires_registration(self, db, scalar_class):
        with pytest.raises(NotRegisteredError):
            db.list_versions(scalar_class, subject=1)


class TestGlobalDatabaseIntegration:
    """Test using global database configuration."""

    def test_save_with_global_db(self, configured_db, scalar_class):
        configured_db.register(scalar_class)
        record_id = scalar_class.save(42, subject=1)  # No db= argument
        assert isinstance(record_id, str)

    def test_load_with_global_db(self, configured_db, scalar_class):
        configured_db.register(scalar_class)
        scalar_class.save(42, subject=1)

        loaded = scalar_class.load(subject=1)  # No db= argument
        assert loaded.data == 42


class TestComplexMetadata:
    """Test with various metadata types."""

    def test_string_metadata(self, db, scalar_class):
        db.register(scalar_class)
        scalar_class.save(42, db=db, name="test_name")
        loaded = scalar_class.load(db=db, name="test_name")
        assert loaded.data == 42

    def test_integer_metadata(self, db, scalar_class):
        db.register(scalar_class)
        scalar_class.save(42, db=db, count=100)
        loaded = scalar_class.load(db=db, count=100)
        assert loaded.data == 42

    def test_float_metadata(self, db, scalar_class):
        db.register(scalar_class)
        scalar_class.save(42, db=db, threshold=0.5)
        loaded = scalar_class.load(db=db, threshold=0.5)
        assert loaded.data == 42

    def test_boolean_metadata(self, db, scalar_class):
        db.register(scalar_class)
        scalar_class.save(42, db=db, active=True)
        loaded = scalar_class.load(db=db, active=True)
        assert loaded.data == 42

    def test_mixed_metadata(self, db, scalar_class):
        db.register(scalar_class)
        scalar_class.save(
            42,
            db=db,
            subject=1,
            name="test",
            threshold=0.5,
            active=True
        )
        loaded = scalar_class.load(
            db=db,
            subject=1,
            name="test",
            threshold=0.5,
            active=True
        )
        assert loaded.data == 42
