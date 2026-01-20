"""Integration tests for scidb - full end-to-end workflows."""

import numpy as np
import pandas as pd
import pytest

from scidb import (
    BaseVariable,
    DatabaseManager,
    NotFoundError,
    configure_database,
)


class TestEndToEndScalarWorkflow:
    """Test complete workflow with scalar values."""

    def test_save_and_load_single_scalar(self, db, scalar_class):
        """Save a scalar and load it back."""
        db.register(scalar_class)

        # Save
        original_value = 42
        vhash = scalar_class.save(original_value, db=db, subject=1, trial=1)

        # Load
        loaded = scalar_class.load(db=db, subject=1, trial=1)

        # Verify
        assert loaded.data == original_value
        assert loaded.vhash == vhash
        assert loaded.metadata == {"subject": 1, "trial": 1}

    def test_multiple_subjects_and_trials(self, db, scalar_class):
        """Save and load data for multiple subjects and trials."""
        db.register(scalar_class)

        # Save data for 3 subjects, 2 trials each
        expected_data = {}
        for subject in range(1, 4):
            for trial in range(1, 3):
                value = subject * 10 + trial
                scalar_class.save(value, db=db, subject=subject, trial=trial)
                expected_data[(subject, trial)] = value

        # Load and verify each
        for (subject, trial), expected_value in expected_data.items():
            loaded = scalar_class.load(db=db, subject=subject, trial=trial)
            assert loaded.data == expected_value

    def test_version_history(self, db, scalar_class):
        """Test that version history is maintained."""
        db.register(scalar_class)

        # Save multiple versions with different data
        vhash1 = scalar_class.save(100, db=db, subject=1, trial=1)
        vhash2 = scalar_class.save(200, db=db, subject=1, trial=1)
        vhash3 = scalar_class.save(300, db=db, subject=1, trial=1)

        # All vhashes should be different
        assert len({vhash1, vhash2, vhash3}) == 3

        # List versions should show all three
        versions = db.list_versions(scalar_class, subject=1, trial=1)
        assert len(versions) == 3

        # Should be able to load specific version
        loaded = scalar_class.load(db=db, version=vhash2)
        assert loaded.data == 200


class TestEndToEndArrayWorkflow:
    """Test complete workflow with numpy arrays."""

    def test_save_and_load_1d_array(self, db, array_class):
        """Save and load a 1D array."""
        db.register(array_class)

        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        array_class.save(original, db=db, subject=1, measurement="signal")

        loaded = array_class.load(db=db, subject=1, measurement="signal")
        np.testing.assert_array_equal(loaded.data, original)

    def test_save_and_load_2d_array(self, db, matrix_class):
        """Save and load a 2D array."""
        db.register(matrix_class)

        original = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        matrix_class.save(original, db=db, subject=1, type="rotation")

        loaded = matrix_class.load(db=db, subject=1, type="rotation")
        np.testing.assert_array_equal(loaded.data, original)

    def test_save_and_load_large_array(self, db, array_class):
        """Save and load a larger array."""
        db.register(array_class)

        original = np.random.rand(10000)
        array_class.save(original, db=db, subject=1, type="timeseries")

        loaded = array_class.load(db=db, subject=1, type="timeseries")
        np.testing.assert_array_almost_equal(loaded.data, original)

    def test_preserve_dtype(self, db, array_class):
        """Test that array dtype is preserved."""
        db.register(array_class)

        for dtype in [np.int32, np.int64, np.float32, np.float64]:
            original = np.array([1, 2, 3], dtype=dtype)
            array_class.save(original, db=db, subject=1, dtype=str(dtype))

            loaded = array_class.load(db=db, subject=1, dtype=str(dtype))
            assert loaded.data.dtype == dtype


class TestEndToEndDataFrameWorkflow:
    """Test complete workflow with pandas DataFrames."""

    def test_save_and_load_dataframe(self, db, dataframe_class):
        """Save and load a DataFrame."""
        db.register(dataframe_class)

        original = pd.DataFrame({
            "time": [0.0, 0.1, 0.2, 0.3],
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [5.0, 6.0, 7.0, 8.0],
        })
        dataframe_class.save(original, db=db, subject=1, trial=1)

        loaded = dataframe_class.load(db=db, subject=1, trial=1)
        pd.testing.assert_frame_equal(loaded.data, original)

    def test_preserve_column_types(self, db, dataframe_class):
        """Test that column types are preserved."""
        db.register(dataframe_class)

        original = pd.DataFrame({
            "int_col": pd.array([1, 2, 3], dtype="int64"),
            "float_col": pd.array([1.1, 2.2, 3.3], dtype="float64"),
            "str_col": ["a", "b", "c"],
        })
        dataframe_class.save(original, db=db, subject=1)

        loaded = dataframe_class.load(db=db, subject=1)
        for col in original.columns:
            assert loaded.data[col].dtype == original[col].dtype


class TestIdempotentSaves:
    """Test that saves are idempotent."""

    def test_same_data_same_metadata_same_vhash(self, db, scalar_class):
        """Saving identical data+metadata should return same vhash."""
        db.register(scalar_class)

        vhash1 = scalar_class.save(42, db=db, subject=1, trial=1)
        vhash2 = scalar_class.save(42, db=db, subject=1, trial=1)
        vhash3 = scalar_class.save(42, db=db, subject=1, trial=1)

        assert vhash1 == vhash2 == vhash3

        # Should only have one row in database
        cursor = db.connection.execute("SELECT COUNT(*) FROM scalar_value")
        assert cursor.fetchone()[0] == 1

    def test_same_array_data_same_vhash(self, db, array_class):
        """Saving identical array data should return same vhash."""
        db.register(array_class)

        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0, 3.0])

        vhash1 = array_class.save(arr1, db=db, subject=1)
        vhash2 = array_class.save(arr2, db=db, subject=1)

        assert vhash1 == vhash2


class TestMultipleVariableTypes:
    """Test working with multiple variable types simultaneously."""

    def test_register_and_use_multiple_types(
        self, db, scalar_class, array_class, matrix_class
    ):
        """Register and use multiple variable types."""
        db.register(scalar_class)
        db.register(array_class)
        db.register(matrix_class)

        # Save different types
        scalar_class.save(42, db=db, subject=1, type="scalar")
        array_class.save(np.array([1, 2, 3]), db=db, subject=1, type="array")
        matrix_class.save(np.eye(3), db=db, subject=1, type="matrix")

        # Load each type
        scalar = scalar_class.load(db=db, subject=1, type="scalar")
        array = array_class.load(db=db, subject=1, type="array")
        matrix = matrix_class.load(db=db, subject=1, type="matrix")

        assert scalar.data == 42
        np.testing.assert_array_equal(array.data, [1, 2, 3])
        np.testing.assert_array_equal(matrix.data, np.eye(3))

    def test_same_metadata_different_types(self, db, scalar_class, array_class):
        """Same metadata can be used for different types."""
        db.register(scalar_class)
        db.register(array_class)

        # Save with same metadata but different types
        scalar_class.save(42, db=db, subject=1, trial=1)
        array_class.save(np.array([1, 2, 3]), db=db, subject=1, trial=1)

        # Load each type specifically
        scalar = scalar_class.load(db=db, subject=1, trial=1)
        array = array_class.load(db=db, subject=1, trial=1)

        assert scalar.data == 42
        np.testing.assert_array_equal(array.data, [1, 2, 3])


class TestDatabasePersistence:
    """Test that data persists across database reconnections."""

    def test_data_persists_after_reconnect(self, temp_db_path, scalar_class):
        """Data should persist after closing and reopening database."""
        # First connection - save data
        db1 = DatabaseManager(temp_db_path)
        db1.register(scalar_class)
        vhash = scalar_class.save(42, db=db1, subject=1, trial=1)
        db1.close()

        # Second connection - load data
        db2 = DatabaseManager(temp_db_path)
        db2.register(scalar_class)
        loaded = scalar_class.load(db=db2, subject=1, trial=1)
        db2.close()

        assert loaded.data == 42
        assert loaded.vhash == vhash

    def test_multiple_types_persist(
        self, temp_db_path, scalar_class, array_class
    ):
        """Multiple types should persist after reconnect."""
        # First connection
        db1 = DatabaseManager(temp_db_path)
        db1.register(scalar_class)
        db1.register(array_class)
        scalar_class.save(42, db=db1, subject=1)
        array_class.save(np.array([1, 2, 3]), db=db1, subject=1)
        db1.close()

        # Second connection
        db2 = DatabaseManager(temp_db_path)
        db2.register(scalar_class)
        db2.register(array_class)

        scalar = scalar_class.load(db=db2, subject=1)
        array = array_class.load(db=db2, subject=1)
        db2.close()

        assert scalar.data == 42
        np.testing.assert_array_equal(array.data, [1, 2, 3])


class TestErrorHandling:
    """Test error handling in various scenarios."""

    def test_load_nonexistent_raises_not_found(self, db, scalar_class):
        """Loading nonexistent data should raise NotFoundError."""
        db.register(scalar_class)
        with pytest.raises(NotFoundError):
            scalar_class.load(db=db, subject=999, trial=999)

    def test_load_wrong_type_returns_empty(self, db, scalar_class, array_class):
        """Loading with wrong type should not find data from other type."""
        db.register(scalar_class)
        db.register(array_class)

        scalar_class.save(42, db=db, subject=1, trial=1)

        # Try to load as array - should not find it
        with pytest.raises(NotFoundError):
            array_class.load(db=db, subject=1, trial=1)


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

        db.register(Point3D)

        # Save a point
        original = (1.0, 2.0, 3.0)
        Point3D.save(original, db=db, name="origin")

        # Load it back
        loaded = Point3D.load(db=db, name="origin")
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

        db.register(Config)

        original = {
            "learning_rate": 0.001,
            "layers": [64, 128, 64],
            "activation": "relu",
            "nested": {"a": 1, "b": 2},
        }
        Config.save(original, db=db, experiment="test")

        loaded = Config.load(db=db, experiment="test")
        assert loaded.data == original
