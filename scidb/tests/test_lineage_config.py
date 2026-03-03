"""Tests for lineage_mode configuration — no Thunk required."""

import pytest

from scidb import configure_database
from scidb.database import DatabaseManager

from conftest import DEFAULT_TEST_SCHEMA_KEYS


class TestLineageModeConfiguration:
    """Test lineage_mode configuration."""

    def test_default_lineage_mode_is_strict(self, tmp_path):
        """Default lineage mode should be 'strict'."""
        db = DatabaseManager(tmp_path / "test.db", dataset_schema_keys=DEFAULT_TEST_SCHEMA_KEYS)
        assert db.lineage_mode == "strict"
        db.close()

    def test_configure_strict_mode(self, tmp_path):
        """Can explicitly configure strict mode."""
        db = DatabaseManager(tmp_path / "test.db", dataset_schema_keys=DEFAULT_TEST_SCHEMA_KEYS, lineage_mode="strict")
        assert db.lineage_mode == "strict"
        db.close()

    def test_configure_ephemeral_mode(self, tmp_path):
        """Can configure ephemeral mode."""
        db = DatabaseManager(tmp_path / "test.db", dataset_schema_keys=DEFAULT_TEST_SCHEMA_KEYS, lineage_mode="ephemeral")
        assert db.lineage_mode == "ephemeral"
        db.close()

    def test_invalid_lineage_mode_raises_error(self, tmp_path):
        """Invalid lineage mode should raise ValueError."""
        with pytest.raises(ValueError, match="lineage_mode must be one of"):
            DatabaseManager(tmp_path / "test.db", dataset_schema_keys=DEFAULT_TEST_SCHEMA_KEYS, lineage_mode="invalid")

    def test_configure_database_passes_lineage_mode(self, tmp_path):
        """configure_database should pass lineage_mode to DatabaseManager."""
        db = configure_database(tmp_path / "test.db", DEFAULT_TEST_SCHEMA_KEYS, lineage_mode="ephemeral")
        assert db.lineage_mode == "ephemeral"
        db.close()
