"""Integration tests for scidb - full end-to-end workflows."""

import numpy as np
import pandas as pd
import pytest

from scidb import (
    BaseVariable,
    NotFoundError,
    configure_database,
)

from conftest import DEFAULT_TEST_SCHEMA_KEYS

class TestEndToEndScalarWorkflow:
    """Test complete workflow with scalar values."""

    def test_save_and_load_single_scalar(self, db, scalar_class):
        """Save a scalar and load it back."""
        # Save
        original_value = 42
        record_id = scalar_class.save(original_value, subject=1, trial=1)

        # Load
        loaded = scalar_class.load(subject=1, trial=1)

        # Verify
        assert loaded.data == original_value
        assert loaded.record_id == record_id
        assert loaded.metadata == {"subject": 1, "trial": 1}

    def test_multiple_subjects_and_trials(self, db, scalar_class):
        """Save and load data for multiple subjects and trials."""
        # Save data for 3 subjects, 2 trials each
        expected_data = {}
        for subject in range(1, 4):
            for trial in range(1, 3):
                value = subject * 10 + trial
                scalar_class.save(value, subject=subject, trial=trial)
                expected_data[(subject, trial)] = value

        # Load and verify each
        for (subject, trial), expected_value in expected_data.items():
            loaded = scalar_class.load(subject=subject, trial=trial)
            assert loaded.data == expected_value

    def test_version_history(self, db, scalar_class):
        """Test that version history is maintained."""
        # Save multiple versions with different data
        record_id1 = scalar_class.save(100, subject=1, trial=1)
        record_id2 = scalar_class.save(200, subject=1, trial=1)
        record_id3 = scalar_class.save(300, subject=1, trial=1)

        # All record_ides should be different
        assert len({record_id1, record_id2, record_id3}) == 3

        # List versions should show all three
        versions = db.list_versions(scalar_class, subject=1, trial=1)
        assert len(versions) == 3

        # Should be able to load specific version
        loaded = scalar_class.load(version=record_id2)
        assert loaded.data == 200


class TestEndToEndArrayWorkflow:
    """Test complete workflow with numpy arrays."""

    def test_save_and_load_1d_array(self, db, array_class):
        """Save and load a 1D array."""
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        array_class.save(original, subject=1, measurement="signal")

        loaded = array_class.load(subject=1, measurement="signal")
        np.testing.assert_array_equal(loaded.data, original)

    def test_save_and_load_2d_array(self, db, matrix_class):
        """Save and load a 2D array."""
        original = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        matrix_class.save(original, subject=1, type="rotation")

        loaded = matrix_class.load(subject=1, type="rotation")
        np.testing.assert_array_equal(loaded.data, original)

    def test_save_and_load_large_array(self, db, array_class):
        """Save and load a larger array."""
        original = np.random.rand(10000)
        array_class.save(original, subject=1, type="timeseries")

        loaded = array_class.load(subject=1, type="timeseries")
        np.testing.assert_array_almost_equal(loaded.data, original)

    def test_preserve_dtype(self, db, array_class):
        """Test that array dtype is preserved."""
        for dtype in [np.int32, np.int64, np.float32, np.float64]:
            original = np.array([1, 2, 3], dtype=dtype)
            array_class.save(original, subject=1, dtype=str(dtype))

            loaded = array_class.load(subject=1, dtype=str(dtype))
            assert loaded.data.dtype == dtype


class TestEndToEndDataFrameWorkflow:
    """Test complete workflow with pandas DataFrames."""

    def test_save_and_load_dataframe(self, db, dataframe_class):
        """Save and load a DataFrame."""
        original = pd.DataFrame({
            "time": [0.0, 0.1, 0.2, 0.3],
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [5.0, 6.0, 7.0, 8.0],
        })
        dataframe_class.save(original, subject=1, trial=1)

        loaded = dataframe_class.load(subject=1, trial=1)
        pd.testing.assert_frame_equal(loaded.data, original)

    def test_preserve_column_types(self, db, dataframe_class):
        """Test that column types are preserved."""
        original = pd.DataFrame({
            "int_col": pd.array([1, 2, 3], dtype="int64"),
            "float_col": pd.array([1.1, 2.2, 3.3], dtype="float64"),
            "str_col": ["a", "b", "c"],
        })
        dataframe_class.save(original, subject=1)

        loaded = dataframe_class.load(subject=1)
        for col in original.columns:
            assert loaded.data[col].dtype == original[col].dtype


class TestIdempotentSaves:
    """Test that saves are idempotent."""

    def test_same_data_same_metadata_same_record_id(self, db, scalar_class):
        """Saving identical data+metadata should return same record_id."""
        record_id1 = scalar_class.save(42, subject=1, trial=1)
        record_id2 = scalar_class.save(42, subject=1, trial=1)
        record_id3 = scalar_class.save(42, subject=1, trial=1)

        assert record_id1 == record_id2 == record_id3

        # Should only have one unique record_id in database
        rows = db._duck._fetchall("SELECT DISTINCT _record_id FROM scalar_value")
        assert len(rows) == 1

    def test_same_array_data_same_record_id(self, db, array_class):
        """Saving identical array data should return same record_id."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0, 3.0])

        record_id1 = array_class.save(arr1, subject=1)
        record_id2 = array_class.save(arr2, subject=1)

        assert record_id1 == record_id2


class TestMultipleVariableTypes:
    """Test working with multiple variable types simultaneously."""

    def test_register_and_use_multiple_types(
        self, db, scalar_class, array_class, matrix_class
    ):
        """Register and use multiple variable types."""
        # Save different types
        scalar_class.save(42, subject=1, type="scalar")
        array_class.save(np.array([1, 2, 3]), subject=1, type="array")
        matrix_class.save(np.eye(3), subject=1, type="matrix")

        # Load each type
        scalar = scalar_class.load(subject=1, type="scalar")
        array = array_class.load(subject=1, type="array")
        matrix = matrix_class.load(subject=1, type="matrix")

        assert scalar.data == 42
        np.testing.assert_array_equal(array.data, [1, 2, 3])
        np.testing.assert_array_equal(matrix.data, np.eye(3))

    def test_same_metadata_different_types(self, db, scalar_class, array_class):
        """Same metadata can be used for different types."""
        # Save with same metadata but different types
        scalar_class.save(42, subject=1, trial=1)
        array_class.save(np.array([1, 2, 3]), subject=1, trial=1)

        # Load each type specifically
        scalar = scalar_class.load(subject=1, trial=1)
        array = array_class.load(subject=1, trial=1)

        assert scalar.data == 42
        np.testing.assert_array_equal(array.data, [1, 2, 3])


class TestDatabasePersistence:
    """Test that data persists across database reconnections."""

    def test_data_persists_after_reconnect(self, tmp_path, scalar_class):
        """Data should persist after closing and reopening database."""
        db_path = tmp_path / "persist_test.duckdb"
        pipeline_path = tmp_path / "persist_pipeline.db"

        # First connection - save data
        db1 = configure_database(db_path, DEFAULT_TEST_SCHEMA_KEYS, pipeline_path)
        record_id = scalar_class.save(42, subject=1, trial=1)
        db1.close()

        # Second connection - load data
        db2 = configure_database(db_path, DEFAULT_TEST_SCHEMA_KEYS, pipeline_path)
        loaded = scalar_class.load(subject=1, trial=1)
        db2.close()

        assert loaded.data == 42
        assert loaded.record_id == record_id

    def test_multiple_types_persist(
        self, tmp_path, scalar_class, array_class
    ):
        """Multiple types should persist after reconnect."""
        db_path = tmp_path / "persist_test.duckdb"
        pipeline_path = tmp_path / "persist_pipeline.db"

        # First connection
        db1 = configure_database(db_path, DEFAULT_TEST_SCHEMA_KEYS, pipeline_path)
        scalar_class.save(42, subject=1)
        array_class.save(np.array([1, 2, 3]), subject=1)
        db1.close()

        # Second connection
        db2 = configure_database(db_path, DEFAULT_TEST_SCHEMA_KEYS, pipeline_path)

        scalar = scalar_class.load(subject=1)
        array = array_class.load(subject=1)
        db2.close()

        assert scalar.data == 42
        np.testing.assert_array_equal(array.data, [1, 2, 3])


class TestErrorHandling:
    """Test error handling in various scenarios."""

    def test_load_nonexistent_raises_not_found(self, db, scalar_class):
        """Loading nonexistent data should raise NotFoundError."""
        with pytest.raises(NotFoundError):
            scalar_class.load(subject=999, trial=999)

    def test_load_wrong_type_returns_empty(self, db, scalar_class, array_class):
        """Loading with wrong type should not find data from other type."""
        scalar_class.save(42, subject=1, trial=1)

        # Try to load as array - should not find it
        with pytest.raises(NotFoundError):
            array_class.load(subject=1, trial=1)


class TestCustomVariableType:
    """Test creating and using custom variable types."""

    def test_custom_variable_type(self, db):
        """Test a custom variable type with complex serialization."""

        class Point3D(BaseVariable):
            """Represents a 3D point."""
            schema_version = 1

            def to_db(self) -> pd.DataFrame:
                x, y, z = self.data
                return pd.DataFrame({"x": [x], "y": [y], "z": [z]})

            @classmethod
            def from_db(cls, df: pd.DataFrame) -> tuple:
                row = df.iloc[0]
                return (row["x"], row["y"], row["z"])

        # Save a point
        original = (1.0, 2.0, 3.0)
        Point3D.save(original, name="origin")

        # Load it back
        loaded = Point3D.load(name="origin")
        assert loaded.data == original

    def test_variable_with_nested_data(self, db):
        """Test a variable type with nested data structure."""

        class Config(BaseVariable):
            """Represents a configuration dict."""
            schema_version = 1

            def to_db(self) -> pd.DataFrame:
                import json
                return pd.DataFrame({"config_json": [json.dumps(self.data)]})

            @classmethod
            def from_db(cls, df: pd.DataFrame) -> dict:
                import json
                return json.loads(df["config_json"].iloc[0])

        original = {
            "learning_rate": 0.001,
            "layers": [64, 128, 64],
            "activation": "relu",
            "nested": {"a": 1, "b": 2},
        }
        Config.save(original, experiment="test")

        loaded = Config.load(experiment="test")
        assert loaded.data == original
