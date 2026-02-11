"""End-to-end integration tests for SciDuck."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sciduck import SciDuck


@pytest.fixture
def duck(tmp_path):
    """Provide a fresh in-memory SciDuck instance."""
    db = SciDuck(":memory:", dataset_schema=["subject", "session", "trial"])
    yield db
    db.close()


@pytest.fixture
def file_duck(tmp_path):
    """Provide a SciDuck instance backed by a file."""
    db_path = tmp_path / "test.duckdb"
    db = SciDuck(db_path, dataset_schema=["subject", "session"])
    yield db
    db.close()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    """Test SciDuck initialization."""

    def test_creates_metadata_tables(self, duck):
        """Should create _schema, _variables, and _variable_groups tables."""
        tables = [
            row[0]
            for row in duck._fetchall(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            )
        ]
        assert "_schema" in tables
        assert "_variables" in tables
        assert "_variable_groups" in tables

    def test_stores_dataset_schema(self, duck):
        """Should store the dataset schema."""
        assert duck.dataset_schema == ["subject", "session", "trial"]

    def test_schema_mismatch_raises(self, tmp_path):
        """Reopening with different schema should raise ValueError."""
        db_path = tmp_path / "schema_test.duckdb"
        db1 = SciDuck(db_path, dataset_schema=["subject", "session"])
        # Save something to populate _schema
        db1.save("var1", 42, subject="S01", session="A")
        db1.close()

        with pytest.raises(ValueError, match="schema mismatch"):
            SciDuck(db_path, dataset_schema=["subject", "trial"])

    def test_repr(self, duck):
        """Should return a readable repr."""
        r = repr(duck)
        assert "SciDuck" in r
        assert "variables=" in r

    def test_context_manager(self, tmp_path):
        """Should work as a context manager."""
        db_path = tmp_path / "ctx.duckdb"
        with SciDuck(db_path, dataset_schema=["subject"]) as db:
            db.save("x", 1, subject="S01")
        # Connection should be closed after exiting


# ---------------------------------------------------------------------------
# Save / Load — Mode B (single entry via kwargs)
# ---------------------------------------------------------------------------

class TestSaveLoadModeB:
    """Test save/load with single entry via schema kwargs."""

    def test_save_and_load_int(self, duck):
        """Round-trip an integer."""
        duck.save("my_int", 42, subject="S01", session="A", trial="1")
        loaded = duck.load("my_int", subject="S01", session="A", trial="1")
        assert loaded == 42

    def test_save_and_load_float(self, duck):
        """Round-trip a float."""
        duck.save("my_float", 3.14, subject="S01", session="A", trial="1")
        loaded = duck.load("my_float", subject="S01", session="A", trial="1")
        assert abs(loaded - 3.14) < 1e-10

    def test_save_and_load_string(self, duck):
        """Round-trip a string."""
        duck.save("my_str", "hello world", subject="S01", session="A", trial="1")
        loaded = duck.load("my_str", subject="S01", session="A", trial="1")
        assert loaded == "hello world"

    def test_save_and_load_bool(self, duck):
        """Round-trip a boolean."""
        duck.save("my_bool", True, subject="S01", session="A", trial="1")
        loaded = duck.load("my_bool", subject="S01", session="A", trial="1")
        assert loaded is True

    def test_save_and_load_1d_array(self, duck):
        """Round-trip a 1D numpy array."""
        original = np.array([1.0, 2.0, 3.0, 4.0])
        duck.save("arr1d", original, subject="S01", session="A", trial="1")
        loaded = duck.load("arr1d", subject="S01", session="A", trial="1")
        np.testing.assert_array_almost_equal(loaded, original)

    def test_save_and_load_2d_array(self, duck):
        """Round-trip a 2D numpy array."""
        original = np.array([[1, 2, 3], [4, 5, 6]])
        duck.save("arr2d", original, subject="S01", session="A", trial="1")
        loaded = duck.load("arr2d", subject="S01", session="A", trial="1")
        np.testing.assert_array_equal(loaded, original)

    def test_save_and_load_list_of_floats(self, duck):
        """Round-trip a list of floats."""
        original = [1.0, 2.5, 3.7]
        duck.save("my_list", original, subject="S01", session="A", trial="1")
        loaded = duck.load("my_list", subject="S01", session="A", trial="1")
        assert loaded == pytest.approx(original)

    def test_save_and_load_list_of_strings(self, duck):
        """Round-trip a list of strings."""
        original = ["alpha", "beta", "gamma"]
        duck.save("str_list", original, subject="S01", session="A", trial="1")
        loaded = duck.load("str_list", subject="S01", session="A", trial="1")
        assert list(loaded) == original

    def test_save_and_load_dict(self, duck):
        """Round-trip a dict (stored as JSON)."""
        original = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}
        duck.save("my_dict", original, subject="S01", session="A", trial="1")
        loaded = duck.load("my_dict", subject="S01", session="A", trial="1")
        assert loaded["key1"] == "value1"
        assert loaded["key2"] == 42
        assert loaded["key3"] == [1, 2, 3]

    def test_save_and_load_dict_with_ndarray(self, duck):
        """Round-trip a dict containing numpy arrays."""
        original = {"weights": np.array([0.1, 0.2, 0.3]), "bias": 0.5}
        duck.save("params", original, subject="S01", session="A", trial="1")
        loaded = duck.load("params", subject="S01", session="A", trial="1")
        np.testing.assert_array_almost_equal(loaded["weights"], original["weights"])
        assert loaded["bias"] == 0.5


# ---------------------------------------------------------------------------
# Save / Load — Mode A (DataFrame with schema columns)
# ---------------------------------------------------------------------------

class TestSaveLoadModeA:
    """Test save/load with DataFrame containing schema columns."""

    def test_save_dataframe_with_schema_columns(self, duck):
        """Save a DataFrame that has subject/session/trial columns."""
        df = pd.DataFrame({
            "subject": ["S01", "S01", "S02", "S02"],
            "session": ["A", "A", "A", "A"],
            "trial": ["1", "2", "1", "2"],
            "value": [10, 20, 30, 40],
        })
        vid = duck.save("df_var", df)

        # Load specific entry
        loaded = duck.load("df_var", subject="S01", session="A", trial="1")
        assert loaded == 10

    def test_save_multi_column_dataframe(self, duck):
        """Save DataFrame with multiple data columns."""
        df = pd.DataFrame({
            "subject": ["S01", "S02"],
            "session": ["A", "A"],
            "trial": ["1", "1"],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
        })
        duck.save("multi_col", df)

        loaded_df = duck.load("multi_col", raw=False)
        assert len(loaded_df) == 2
        assert "x" in loaded_df.columns
        assert "y" in loaded_df.columns


# ---------------------------------------------------------------------------
# Save / Load — Mode C (dict with tuple keys)
# ---------------------------------------------------------------------------

class TestSaveLoadModeC:
    """Test save/load with dict mapping tuples to values."""

    def test_save_dict_with_tuple_keys(self, duck):
        """Save a dict mapping (subject, session, trial) tuples to values."""
        data = {
            ("S01", "A", "1"): 100,
            ("S01", "A", "2"): 200,
            ("S02", "A", "1"): 300,
        }
        duck.save("tuple_dict", data)

        loaded = duck.load("tuple_dict", subject="S01", session="A", trial="1")
        assert loaded == 100

        loaded = duck.load("tuple_dict", subject="S02", session="A", trial="1")
        assert loaded == 300


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------

class TestVersioning:
    """Test version management features."""

    def test_auto_increment_version(self, duck):
        """Save multiple versions, version_id should auto-increment."""
        v1 = duck.save("versioned", 10, subject="S01", session="A", trial="1", force=True)
        v2 = duck.save("versioned", 20, subject="S01", session="A", trial="1", force=True)
        v3 = duck.save("versioned", 30, subject="S01", session="A", trial="1", force=True)

        assert v1 == 1
        assert v2 == 2
        assert v3 == 3

    def test_latest_version_loaded_by_default(self, duck):
        """Load should return the latest version by default."""
        duck.save("versioned", 10, subject="S01", session="A", trial="1", force=True)
        duck.save("versioned", 20, subject="S01", session="A", trial="1", force=True)
        duck.save("versioned", 30, subject="S01", session="A", trial="1", force=True)

        loaded = duck.load("versioned", subject="S01", session="A", trial="1")
        assert loaded == 30

    def test_load_specific_version(self, duck):
        """Load a specific version by version_id."""
        duck.save("versioned", 10, subject="S01", session="A", trial="1", force=True)
        duck.save("versioned", 20, subject="S01", session="A", trial="1", force=True)

        loaded = duck.load("versioned", version_id=1, subject="S01", session="A", trial="1")
        assert loaded == 10

    def test_list_versions(self, duck):
        """Should list all versions of a variable."""
        duck.save("versioned", 10, subject="S01", session="A", trial="1", force=True)
        duck.save("versioned", 20, subject="S01", session="A", trial="1", force=True)

        versions = duck.list_versions("versioned")
        assert len(versions) == 2

    def test_saving_same_data_creates_new_version(self, duck):
        """Saving identical data creates a new version (no hash dedup)."""
        v1 = duck.save("dedup", 42, subject="S01", session="A", trial="1")
        v2 = duck.save("dedup", 42, subject="S01", session="A", trial="1")

        assert v2 == v1 + 1  # New version created each time


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

class TestDelete:
    """Test delete functionality."""

    def test_delete_variable(self, duck):
        """Should delete a variable and all its versions."""
        duck.save("to_delete", 42, subject="S01", session="A", trial="1")
        duck.delete("to_delete")

        # Variable table should be gone
        assert not duck._table_exists("to_delete")

        # Should not appear in list_variables
        var_list = duck.list_variables()
        assert len(var_list[var_list["variable_name"] == "to_delete"]) == 0

    def test_delete_specific_version(self, duck):
        """Should delete only the specified version."""
        duck.save("multi_ver", 10, subject="S01", session="A", trial="1", force=True)
        duck.save("multi_ver", 20, subject="S01", session="A", trial="1", force=True)

        duck.delete("multi_ver", version_id=1)

        versions = duck.list_versions("multi_ver")
        assert len(versions) == 1
        assert versions["version_id"].iloc[0] == 2


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------

class TestGroups:
    """Test variable grouping functionality."""

    def test_add_to_group(self, duck):
        """Should add variables to a group."""
        duck.save("var_a", 1, subject="S01", session="A", trial="1")
        duck.save("var_b", 2, subject="S01", session="A", trial="1")

        duck.add_to_group("my_group", ["var_a", "var_b"])

        members = duck.get_group("my_group")
        assert "var_a" in members
        assert "var_b" in members

    def test_add_single_string(self, duck):
        """Should accept a single string instead of list."""
        duck.add_to_group("single", "var_x")
        members = duck.get_group("single")
        assert "var_x" in members

    def test_remove_from_group(self, duck):
        """Should remove variable from group."""
        duck.add_to_group("grp", ["var_a", "var_b"])
        duck.remove_from_group("grp", "var_a")

        members = duck.get_group("grp")
        assert "var_a" not in members
        assert "var_b" in members

    def test_list_groups(self, duck):
        """Should list all group names."""
        duck.add_to_group("group_1", "a")
        duck.add_to_group("group_2", "b")

        groups = duck.list_groups()
        assert "group_1" in groups
        assert "group_2" in groups

    def test_add_duplicate_is_idempotent(self, duck):
        """Adding same variable to same group twice should not duplicate."""
        duck.add_to_group("grp", "var_a")
        duck.add_to_group("grp", "var_a")

        members = duck.get_group("grp")
        assert members.count("var_a") == 1


# ---------------------------------------------------------------------------
# List / Inspect
# ---------------------------------------------------------------------------

class TestListInspect:
    """Test listing and inspection features."""

    def test_list_variables(self, duck):
        """Should list all variables with metadata."""
        duck.save("alpha", 1, subject="S01", session="A", trial="1")
        duck.save("beta", 2, subject="S01", session="A", trial="1")

        var_list = duck.list_variables()
        assert len(var_list) == 2
        names = list(var_list["variable_name"])
        assert "alpha" in names
        assert "beta" in names

    def test_list_variables_shows_version_count(self, duck):
        """list_variables should show the number of versions."""
        duck.save("counted", 1, subject="S01", session="A", trial="1", force=True)
        duck.save("counted", 2, subject="S01", session="A", trial="1", force=True)

        var_list = duck.list_variables()
        row = var_list[var_list["variable_name"] == "counted"].iloc[0]
        assert row["num_versions"] == 2


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test error handling."""

    def test_load_nonexistent_variable_raises(self, duck):
        """Loading a non-existent variable should raise KeyError."""
        with pytest.raises(KeyError, match="not found"):
            duck.load("nonexistent")

    def test_invalid_schema_level_raises(self, duck):
        """Using invalid schema_level should raise ValueError."""
        with pytest.raises(ValueError, match="not in"):
            duck.save("bad_level", 42, schema_level="nonexistent", subject="S01")

    def test_missing_schema_keys_raises(self, duck):
        """Missing required schema keys should raise ValueError."""
        with pytest.raises(ValueError, match="Missing schema keys"):
            duck.save("incomplete", 42, subject="S01")
            # session and trial missing for the default (lowest) level


# ---------------------------------------------------------------------------
# Dtype Preservation
# ---------------------------------------------------------------------------

class TestDtypePreservation:
    """Test that numpy dtypes are preserved through round-trip."""

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
    def test_1d_array_dtype(self, duck, dtype):
        """1D array dtype should be preserved."""
        original = np.array([1, 2, 3], dtype=dtype)
        duck.save(f"typed_{dtype}", original, subject="S01", session="A", trial="1")
        loaded = duck.load(f"typed_{dtype}", subject="S01", session="A", trial="1")
        assert np.asarray(loaded).dtype == dtype

    def test_2d_array_shape_preserved(self, duck):
        """2D array shape should be preserved."""
        original = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        duck.save("matrix", original, subject="S01", session="A", trial="1")
        loaded = duck.load("matrix", subject="S01", session="A", trial="1")
        assert np.asarray(loaded).shape == (3, 2)


# ---------------------------------------------------------------------------
# Schema Levels
# ---------------------------------------------------------------------------

class TestSchemaLevels:
    """Test saving at different schema levels."""

    def test_save_at_higher_level(self, duck):
        """Should save at subject level (above default trial level)."""
        duck.save(
            "subject_var", 42, schema_level="subject", subject="S01"
        )
        loaded = duck.load("subject_var", subject="S01")
        assert loaded == 42

    def test_save_at_middle_level(self, duck):
        """Should save at session level."""
        duck.save(
            "session_var", 99, schema_level="session", subject="S01", session="A"
        )
        loaded = duck.load("session_var", subject="S01", session="A")
        assert loaded == 99


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    """Test data persistence across connections."""

    def test_data_survives_reconnect(self, tmp_path):
        """Data should be available after close and reopen."""
        db_path = tmp_path / "persist.duckdb"
        schema = ["subject"]

        db1 = SciDuck(db_path, dataset_schema=schema)
        db1.save("persisted", 42, subject="S01")
        db1.close()

        db2 = SciDuck(db_path, dataset_schema=schema)
        loaded = db2.load("persisted", subject="S01")
        db2.close()

        assert loaded == 42

    def test_versions_survive_reconnect(self, tmp_path):
        """Version history should survive reconnection."""
        db_path = tmp_path / "persist_ver.duckdb"
        schema = ["subject"]

        db1 = SciDuck(db_path, dataset_schema=schema)
        db1.save("versioned", 10, subject="S01", force=True)
        db1.save("versioned", 20, subject="S01", force=True)
        db1.close()

        db2 = SciDuck(db_path, dataset_schema=schema)
        versions = db2.list_versions("versioned")
        db2.close()

        assert len(versions) == 2


# ---------------------------------------------------------------------------
# Direct Query
# ---------------------------------------------------------------------------

class TestDirectQuery:
    """Test the query() method for arbitrary SQL."""

    def test_query_returns_dataframe(self, duck):
        """query() should return a pandas DataFrame."""
        duck.save("qvar", 42, subject="S01", session="A", trial="1")
        result = duck.query('SELECT * FROM "qvar"')
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1
