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


class TestForType:
    """Test BaseVariable.for_type() class factory."""

    def test_for_type_creates_subclass(self, scalar_class):
        """for_type() should create a subclass."""
        SpecialScalar = scalar_class.for_type("special")
        assert issubclass(SpecialScalar, scalar_class)

    def test_for_type_has_correct_name(self, scalar_class):
        """Generated class name should include type suffix."""
        SpecialScalar = scalar_class.for_type("temperature")
        assert SpecialScalar.__name__ == "ScalarValueTemperature"

    def test_for_type_has_type_suffix(self, scalar_class):
        """get_type_suffix() should return the suffix."""
        SpecialScalar = scalar_class.for_type("humidity")
        assert SpecialScalar.get_type_suffix() == "humidity"

    def test_for_type_table_name_includes_suffix(self, scalar_class):
        """table_name() should include the type suffix."""
        SpecialScalar = scalar_class.for_type("pressure")
        assert SpecialScalar.table_name() == "scalar_value_pressure"

    def test_for_type_normalizes_spaces(self, scalar_class):
        """Spaces in type name should become underscores."""
        SpecialScalar = scalar_class.for_type("ambient temperature")
        assert SpecialScalar.table_name() == "scalar_value_ambient_temperature"

    def test_for_type_normalizes_hyphens(self, scalar_class):
        """Hyphens in type name should become underscores."""
        SpecialScalar = scalar_class.for_type("air-quality")
        assert SpecialScalar.table_name() == "scalar_value_air_quality"

    def test_for_type_normalizes_uppercase(self, scalar_class):
        """Uppercase in type name should become lowercase."""
        SpecialScalar = scalar_class.for_type("Temperature")
        assert SpecialScalar.table_name() == "scalar_value_temperature"

    def test_for_type_preserves_schema_version(self, scalar_class):
        """Specialized class should inherit schema_version."""
        SpecialScalar = scalar_class.for_type("test")
        assert SpecialScalar.schema_version == scalar_class.schema_version

    def test_for_type_inherits_to_db(self, scalar_class):
        """Specialized class should inherit to_db() method."""
        SpecialScalar = scalar_class.for_type("test")
        instance = SpecialScalar(42)
        df = instance.to_db()
        assert df["value"].iloc[0] == 42

    def test_for_type_inherits_from_db(self, scalar_class):
        """Specialized class should inherit from_db() method."""
        SpecialScalar = scalar_class.for_type("test")
        df = pd.DataFrame({"value": [123]})
        data = SpecialScalar.from_db(df)
        assert data == 123

    def test_multiple_for_type_different_tables(self, scalar_class):
        """Multiple for_type() calls should create different table names."""
        Type1 = scalar_class.for_type("alpha")
        Type2 = scalar_class.for_type("beta")
        Type3 = scalar_class.for_type("gamma")

        assert Type1.table_name() == "scalar_value_alpha"
        assert Type2.table_name() == "scalar_value_beta"
        assert Type3.table_name() == "scalar_value_gamma"

    def test_original_class_unchanged(self, scalar_class):
        """Original class should not be modified by for_type()."""
        _ = scalar_class.for_type("test")
        assert scalar_class.get_type_suffix() is None
        assert scalar_class.table_name() == "scalar_value"

    def test_for_type_works_with_database(self, db, array_class):
        """Specialized types should work with save/load."""
        TempArray = array_class.for_type("temperature")
        HumidArray = array_class.for_type("humidity")

        db.register(TempArray)
        db.register(HumidArray)

        # Save different data to each type
        temp_data = np.array([20.0, 21.0, 22.0])
        humid_data = np.array([60.0, 65.0, 70.0])

        TempArray(temp_data).save(db=db, sensor=1)
        HumidArray(humid_data).save(db=db, sensor=1)

        # Load back - should get correct data for each type
        loaded_temp = TempArray.load(db=db, sensor=1)
        loaded_humid = HumidArray.load(db=db, sensor=1)

        np.testing.assert_array_equal(loaded_temp.data, temp_data)
        np.testing.assert_array_equal(loaded_humid.data, humid_data)

    def test_for_type_separate_tables_in_database(self, db, scalar_class):
        """Each specialized type should create its own table."""
        Type1 = scalar_class.for_type("alpha")
        Type2 = scalar_class.for_type("beta")

        db.register(Type1)
        db.register(Type2)

        # Check tables exist
        cursor = db.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'scalar_value_%'"
        )
        tables = {row[0] for row in cursor.fetchall()}

        assert "scalar_value_alpha" in tables
        assert "scalar_value_beta" in tables

    def test_for_type_data_isolation(self, db, scalar_class):
        """Data saved to one specialized type should not appear in another."""
        Type1 = scalar_class.for_type("type_a")
        Type2 = scalar_class.for_type("type_b")

        db.register(Type1)
        db.register(Type2)

        # Save to Type1 only
        Type1(100).save(db=db, key="test")

        # Should find in Type1
        loaded = Type1.load(db=db, key="test")
        assert loaded.data == 100

        # Should NOT find in Type2
        from scidb.exceptions import NotFoundError
        with pytest.raises(NotFoundError):
            Type2.load(db=db, key="test")

    def test_for_type_with_array_class(self, db, array_class):
        """for_type() should work with array-based variable classes."""
        SignalA = array_class.for_type("signal_a")
        SignalB = array_class.for_type("signal_b")

        db.register(SignalA)
        db.register(SignalB)

        data_a = np.array([1.0, 2.0, 3.0])
        data_b = np.array([10.0, 20.0, 30.0])

        SignalA(data_a).save(db=db, channel=1)
        SignalB(data_b).save(db=db, channel=1)

        # Verify separate storage
        loaded_a = SignalA.load(db=db, channel=1)
        loaded_b = SignalB.load(db=db, channel=1)

        np.testing.assert_array_equal(loaded_a.data, data_a)
        np.testing.assert_array_equal(loaded_b.data, data_b)

    def test_for_type_no_argument_creates_default(self, scalar_class):
        """for_type() with no argument should create a default type."""
        DefaultScalar = scalar_class.for_type()
        assert issubclass(DefaultScalar, scalar_class)
        assert DefaultScalar.__name__ == "ScalarValueDefault"

    def test_for_type_empty_string_creates_default(self, scalar_class):
        """for_type('') should create a default type."""
        DefaultScalar = scalar_class.for_type("")
        assert issubclass(DefaultScalar, scalar_class)
        assert DefaultScalar.__name__ == "ScalarValueDefault"

    def test_default_type_uses_base_table(self, scalar_class):
        """Default type should use the same table as the base class."""
        DefaultScalar = scalar_class.for_type()
        assert DefaultScalar.table_name() == scalar_class.table_name()
        assert DefaultScalar.table_name() == "scalar_value"

    def test_default_type_has_empty_suffix(self, scalar_class):
        """Default type should have empty string as suffix."""
        DefaultScalar = scalar_class.for_type()
        assert DefaultScalar.get_type_suffix() == ""

    def test_default_type_is_distinct_from_base(self, scalar_class):
        """Default type should be a different class from the base."""
        DefaultScalar = scalar_class.for_type()
        assert DefaultScalar is not scalar_class

    def test_migration_one_to_one_to_one_to_many(self, db, scalar_class):
        """Support migration from one-to-one to one-to-many mapping."""
        # Step 1: Start with one-to-one (using base class directly)
        db.register(scalar_class)
        scalar_class(42).save(db=db, key="original")

        # Step 2: Later, decide to use one-to-many
        DefaultScalar = scalar_class.for_type()  # For existing data
        TypeA = scalar_class.for_type("type_a")
        TypeB = scalar_class.for_type("type_b")

        db.register(DefaultScalar)  # Same table as scalar_class
        db.register(TypeA)
        db.register(TypeB)

        # Save new typed data
        TypeA(100).save(db=db, key="new_a")
        TypeB(200).save(db=db, key="new_b")

        # Old data should still be accessible via base class
        loaded_original = scalar_class.load(db=db, key="original")
        assert loaded_original.data == 42

        # Old data should also be accessible via default type (same table)
        loaded_via_default = DefaultScalar.load(db=db, key="original")
        assert loaded_via_default.data == 42

        # New typed data should be in separate tables
        loaded_a = TypeA.load(db=db, key="new_a")
        loaded_b = TypeB.load(db=db, key="new_b")
        assert loaded_a.data == 100
        assert loaded_b.data == 200

        # Typed data should NOT be accessible via base class
        from scidb.exceptions import NotFoundError
        with pytest.raises(NotFoundError):
            scalar_class.load(db=db, key="new_a")

    def test_default_type_and_base_share_data(self, db, scalar_class):
        """Default type and base class should access the same data."""
        db.register(scalar_class)
        DefaultScalar = scalar_class.for_type()
        db.register(DefaultScalar)

        # Save via base class
        scalar_class(123).save(db=db, key="shared")

        # Load via default type
        loaded = DefaultScalar.load(db=db, key="shared")
        assert loaded.data == 123

        # Save via default type
        DefaultScalar(456).save(db=db, key="shared2")

        # Load via base class
        loaded2 = scalar_class.load(db=db, key="shared2")
        assert loaded2.data == 456


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
