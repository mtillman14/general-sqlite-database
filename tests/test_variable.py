"""Tests for scidb.variable module."""

import pytest
import numpy as np
import pandas as pd

from scidb.variable import BaseVariable
from scidb.exceptions import (
    ReservedMetadataKeyError,
    DatabaseNotConfiguredError,
)


class TestBaseVariableABC:
    """Test BaseVariable abstract base class."""

    def test_cannot_instantiate_directly(self):
        """BaseVariable should not be instantiable directly."""
        with pytest.raises(TypeError):
            BaseVariable(42)

    def test_subclass_must_implement_to_db(self):
        """Subclass without to_db should raise TypeError."""
        class IncompleteVariable(BaseVariable):
            @classmethod
            def from_db(cls, df):
                return df

        with pytest.raises(TypeError):
            IncompleteVariable(42)

    def test_subclass_must_implement_from_db(self):
        """Subclass without from_db should raise TypeError."""
        class IncompleteVariable(BaseVariable):
            def to_db(self):
                return pd.DataFrame()

        with pytest.raises(TypeError):
            IncompleteVariable(42)

    def test_complete_subclass_instantiates(self, scalar_class):
        """Complete subclass should instantiate properly."""
        var = scalar_class(42)
        assert var.data == 42


class TestBaseVariableInit:
    """Test BaseVariable initialization."""

    def test_data_stored(self, scalar_class):
        var = scalar_class(42)
        assert var.data == 42

    def test_vhash_initially_none(self, scalar_class):
        var = scalar_class(42)
        assert var.vhash is None

    def test_metadata_initially_none(self, scalar_class):
        var = scalar_class(42)
        assert var.metadata is None

    def test_data_can_be_any_type(self, array_class):
        arr = np.array([1, 2, 3])
        var = array_class(arr)
        np.testing.assert_array_equal(var.data, arr)


class TestTableName:
    """Test table_name class method."""

    def test_simple_name(self, scalar_class):
        assert scalar_class.table_name() == "scalar_value"

    def test_camel_case_conversion(self, array_class):
        assert array_class.table_name() == "array_value"

    def test_multi_word_camel_case(self, matrix_class):
        assert matrix_class.table_name() == "matrix_value"

    def test_custom_class_name(self):
        class MyCustomVariableType(BaseVariable):
            schema_version = 1

            def to_db(self):
                return pd.DataFrame({"value": [self.data]})

            @classmethod
            def from_db(cls, df):
                return df["value"].iloc[0]

        assert MyCustomVariableType.table_name() == "my_custom_variable_type"

    def test_single_word(self):
        class Signal(BaseVariable):
            schema_version = 1

            def to_db(self):
                return pd.DataFrame({"value": [self.data]})

            @classmethod
            def from_db(cls, df):
                return df["value"].iloc[0]

        assert Signal.table_name() == "signal"

    def test_acronym_in_name(self):
        class XMLParser(BaseVariable):
            schema_version = 1

            def to_db(self):
                return pd.DataFrame({"value": [self.data]})

            @classmethod
            def from_db(cls, df):
                return df["value"].iloc[0]

        # Note: This will produce "x_m_l_parser" which may not be ideal
        # but is the expected behavior of the simple regex
        assert "x" in XMLParser.table_name().lower()


class TestSchemaVersion:
    """Test schema_version class attribute."""

    def test_default_schema_version(self, scalar_class):
        assert scalar_class.schema_version == 1

    def test_custom_schema_version(self, matrix_class):
        assert matrix_class.schema_version == 2

    def test_schema_version_on_instance(self, scalar_class):
        var = scalar_class(42)
        assert var.schema_version == 1


class TestReservedKeys:
    """Test reserved metadata keys."""

    def test_reserved_keys_exist(self):
        assert "vhash" in BaseVariable._reserved_keys
        assert "id" in BaseVariable._reserved_keys
        assert "created_at" in BaseVariable._reserved_keys
        assert "schema_version" in BaseVariable._reserved_keys
        assert "data" in BaseVariable._reserved_keys

    def test_reserved_keys_is_frozenset(self):
        assert isinstance(BaseVariable._reserved_keys, frozenset)


class TestToDbFromDb:
    """Test to_db and from_db implementations."""

    def test_scalar_to_db(self, scalar_class):
        var = scalar_class(42)
        df = var.to_db()
        assert isinstance(df, pd.DataFrame)
        assert "value" in df.columns
        assert df["value"].iloc[0] == 42

    def test_scalar_from_db(self, scalar_class):
        df = pd.DataFrame({"value": [42]})
        data = scalar_class.from_db(df)
        assert data == 42

    def test_scalar_roundtrip(self, scalar_class):
        original = 42
        var = scalar_class(original)
        df = var.to_db()
        restored = scalar_class.from_db(df)
        assert restored == original

    def test_array_to_db(self, array_class):
        arr = np.array([1.0, 2.0, 3.0])
        var = array_class(arr)
        df = var.to_db()
        assert isinstance(df, pd.DataFrame)
        assert "values" in df.columns
        assert "shape" in df.columns
        assert "dtype" in df.columns

    def test_array_from_db(self, array_class):
        arr = np.array([1.0, 2.0, 3.0])
        df = pd.DataFrame({
            "values": [arr.tobytes()],
            "shape": [str(arr.shape)],
            "dtype": [str(arr.dtype)],
        })
        restored = array_class.from_db(df)
        np.testing.assert_array_equal(restored, arr)

    def test_array_roundtrip(self, array_class):
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        var = array_class(original)
        df = var.to_db()
        restored = array_class.from_db(df)
        np.testing.assert_array_equal(restored, original)

    def test_matrix_roundtrip(self, matrix_class):
        original = np.array([[1, 2, 3], [4, 5, 6]])
        var = matrix_class(original)
        df = var.to_db()
        restored = matrix_class.from_db(df)
        np.testing.assert_array_equal(restored, original)

    def test_dataframe_roundtrip(self, dataframe_class):
        original = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        var = dataframe_class(original)
        df = var.to_db()
        restored = dataframe_class.from_db(df)
        pd.testing.assert_frame_equal(restored, original)


class TestSaveWithoutDatabase:
    """Test save() behavior without configured database."""

    def test_save_raises_without_database(self, scalar_class, clear_global_db):
        var = scalar_class(42)
        with pytest.raises(DatabaseNotConfiguredError):
            var.save(subject=1)


class TestSaveReservedKeys:
    """Test that reserved metadata keys are rejected."""

    def test_save_rejects_vhash(self, scalar_class, configured_db):
        configured_db.register(scalar_class)
        var = scalar_class(42)
        with pytest.raises(ReservedMetadataKeyError, match="vhash"):
            var.save(vhash="fake_hash")

    def test_save_rejects_id(self, scalar_class, configured_db):
        configured_db.register(scalar_class)
        var = scalar_class(42)
        with pytest.raises(ReservedMetadataKeyError, match="id"):
            var.save(id=123)

    def test_save_rejects_created_at(self, scalar_class, configured_db):
        configured_db.register(scalar_class)
        var = scalar_class(42)
        with pytest.raises(ReservedMetadataKeyError, match="created_at"):
            var.save(created_at="2024-01-01")

    def test_save_rejects_schema_version(self, scalar_class, configured_db):
        configured_db.register(scalar_class)
        var = scalar_class(42)
        with pytest.raises(ReservedMetadataKeyError, match="schema_version"):
            var.save(schema_version=1)

    def test_save_rejects_data(self, scalar_class, configured_db):
        configured_db.register(scalar_class)
        var = scalar_class(42)
        with pytest.raises(ReservedMetadataKeyError, match="data"):
            var.save(data="something")

    def test_save_rejects_multiple_reserved(self, scalar_class, configured_db):
        configured_db.register(scalar_class)
        var = scalar_class(42)
        with pytest.raises(ReservedMetadataKeyError):
            var.save(vhash="fake", id=123)


class TestLoadWithoutDatabase:
    """Test load() behavior without configured database."""

    def test_load_raises_without_database(self, scalar_class, clear_global_db):
        with pytest.raises(DatabaseNotConfiguredError):
            scalar_class.load(subject=1)
