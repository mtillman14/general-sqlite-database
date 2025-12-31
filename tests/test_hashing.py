"""Tests for scidb.hashing module."""

import pytest
import numpy as np
import pandas as pd

from scidb.hashing import canonical_hash, generate_vhash, _serialize_for_hash


# Module-level class for pickle fallback test (pickle can't serialize local classes)
class _CustomClassForTest:
    """Custom class used to test pickle fallback in canonical_hash."""
    def __init__(self, value):
        self.value = value


class TestCanonicalHashPrimitives:
    """Test canonical_hash with primitive types."""

    def test_hash_none(self):
        h = canonical_hash(None)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_bool_true(self):
        h = canonical_hash(True)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_bool_false(self):
        h = canonical_hash(False)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_bool_different(self):
        assert canonical_hash(True) != canonical_hash(False)

    def test_hash_int(self):
        h = canonical_hash(42)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_int_deterministic(self):
        assert canonical_hash(42) == canonical_hash(42)

    def test_hash_int_different_values(self):
        assert canonical_hash(42) != canonical_hash(43)

    def test_hash_float(self):
        h = canonical_hash(3.14159)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_float_deterministic(self):
        assert canonical_hash(3.14159) == canonical_hash(3.14159)

    def test_hash_string(self):
        h = canonical_hash("hello world")
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_string_deterministic(self):
        assert canonical_hash("test") == canonical_hash("test")

    def test_hash_string_different_values(self):
        assert canonical_hash("hello") != canonical_hash("world")

    def test_hash_empty_string(self):
        h = canonical_hash("")
        assert isinstance(h, str)
        assert len(h) == 16


class TestCanonicalHashCollections:
    """Test canonical_hash with collections."""

    def test_hash_list(self):
        h = canonical_hash([1, 2, 3])
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_list_deterministic(self):
        assert canonical_hash([1, 2, 3]) == canonical_hash([1, 2, 3])

    def test_hash_list_order_matters(self):
        assert canonical_hash([1, 2, 3]) != canonical_hash([3, 2, 1])

    def test_hash_empty_list(self):
        h = canonical_hash([])
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_tuple(self):
        h = canonical_hash((1, 2, 3))
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_tuple_deterministic(self):
        assert canonical_hash((1, 2, 3)) == canonical_hash((1, 2, 3))

    def test_hash_list_vs_tuple_different(self):
        """Lists and tuples should have different hashes."""
        assert canonical_hash([1, 2, 3]) != canonical_hash((1, 2, 3))

    def test_hash_dict(self):
        h = canonical_hash({"a": 1, "b": 2})
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_dict_deterministic(self):
        assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"a": 1, "b": 2})

    def test_hash_dict_order_independent(self):
        """Dict hash should be independent of insertion order."""
        d1 = {"a": 1, "b": 2, "c": 3}
        d2 = {"c": 3, "a": 1, "b": 2}
        assert canonical_hash(d1) == canonical_hash(d2)

    def test_hash_empty_dict(self):
        h = canonical_hash({})
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_nested_structure(self):
        nested = {"list": [1, 2, 3], "tuple": (4, 5), "nested": {"a": 1}}
        h = canonical_hash(nested)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_nested_deterministic(self):
        nested1 = {"list": [1, 2, 3], "nested": {"a": 1}}
        nested2 = {"list": [1, 2, 3], "nested": {"a": 1}}
        assert canonical_hash(nested1) == canonical_hash(nested2)


class TestCanonicalHashNumpy:
    """Test canonical_hash with numpy arrays."""

    def test_hash_1d_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        h = canonical_hash(arr)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_1d_array_deterministic(self):
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0, 3.0])
        assert canonical_hash(arr1) == canonical_hash(arr2)

    def test_hash_1d_array_different_values(self):
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0, 4.0])
        assert canonical_hash(arr1) != canonical_hash(arr2)

    def test_hash_2d_array(self):
        arr = np.array([[1, 2], [3, 4]])
        h = canonical_hash(arr)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_2d_array_deterministic(self):
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([[1, 2], [3, 4]])
        assert canonical_hash(arr1) == canonical_hash(arr2)

    def test_hash_different_shapes(self):
        """Arrays with same values but different shapes should have different hashes."""
        arr1 = np.array([1, 2, 3, 4])
        arr2 = np.array([[1, 2], [3, 4]])
        assert canonical_hash(arr1) != canonical_hash(arr2)

    def test_hash_different_dtypes(self):
        """Arrays with same values but different dtypes should have different hashes."""
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        arr2 = np.array([1, 2, 3], dtype=np.int64)
        assert canonical_hash(arr1) != canonical_hash(arr2)

    def test_hash_float_array(self):
        arr = np.array([1.1, 2.2, 3.3], dtype=np.float64)
        h = canonical_hash(arr)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_empty_array(self):
        arr = np.array([])
        h = canonical_hash(arr)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_3d_array(self):
        arr = np.random.rand(2, 3, 4)
        h = canonical_hash(arr)
        assert isinstance(h, str)
        assert len(h) == 16


class TestCanonicalHashPandas:
    """Test canonical_hash with pandas objects."""

    def test_hash_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        h = canonical_hash(df)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_dataframe_deterministic(self):
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert canonical_hash(df1) == canonical_hash(df2)

    def test_hash_dataframe_different_values(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2, 4]})
        assert canonical_hash(df1) != canonical_hash(df2)

    def test_hash_dataframe_different_columns(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [1, 2, 3]})
        assert canonical_hash(df1) != canonical_hash(df2)

    def test_hash_series(self):
        s = pd.Series([1, 2, 3], name="test")
        h = canonical_hash(s)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_series_deterministic(self):
        s1 = pd.Series([1, 2, 3], name="test")
        s2 = pd.Series([1, 2, 3], name="test")
        assert canonical_hash(s1) == canonical_hash(s2)

    def test_hash_empty_dataframe(self):
        df = pd.DataFrame()
        h = canonical_hash(df)
        assert isinstance(h, str)
        assert len(h) == 16


class TestCanonicalHashFallback:
    """Test canonical_hash fallback to pickle."""

    def test_hash_custom_object(self):
        obj = _CustomClassForTest(42)
        h = canonical_hash(obj)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_custom_object_deterministic(self):
        """Same custom object values should produce same hash."""
        obj1 = _CustomClassForTest(42)
        obj2 = _CustomClassForTest(42)
        assert canonical_hash(obj1) == canonical_hash(obj2)

    def test_hash_custom_object_different_values(self):
        """Different custom object values should produce different hashes."""
        obj1 = _CustomClassForTest(42)
        obj2 = _CustomClassForTest(43)
        assert canonical_hash(obj1) != canonical_hash(obj2)

    def test_hash_set(self):
        """Sets should be hashed via pickle fallback."""
        s = frozenset([1, 2, 3])
        h = canonical_hash(s)
        assert isinstance(h, str)
        assert len(h) == 16


class TestGenerateVhash:
    """Test generate_vhash function."""

    def test_basic_vhash(self):
        vhash = generate_vhash(
            class_name="TestClass",
            schema_version=1,
            data={"value": 42},
            metadata={"subject": 1},
        )
        assert isinstance(vhash, str)
        assert len(vhash) == 16

    def test_vhash_deterministic(self):
        vhash1 = generate_vhash(
            class_name="TestClass",
            schema_version=1,
            data={"value": 42},
            metadata={"subject": 1},
        )
        vhash2 = generate_vhash(
            class_name="TestClass",
            schema_version=1,
            data={"value": 42},
            metadata={"subject": 1},
        )
        assert vhash1 == vhash2

    def test_vhash_different_class_name(self):
        vhash1 = generate_vhash("ClassA", 1, {"value": 42}, {"subject": 1})
        vhash2 = generate_vhash("ClassB", 1, {"value": 42}, {"subject": 1})
        assert vhash1 != vhash2

    def test_vhash_different_schema_version(self):
        vhash1 = generate_vhash("TestClass", 1, {"value": 42}, {"subject": 1})
        vhash2 = generate_vhash("TestClass", 2, {"value": 42}, {"subject": 1})
        assert vhash1 != vhash2

    def test_vhash_different_data(self):
        vhash1 = generate_vhash("TestClass", 1, {"value": 42}, {"subject": 1})
        vhash2 = generate_vhash("TestClass", 1, {"value": 43}, {"subject": 1})
        assert vhash1 != vhash2

    def test_vhash_different_metadata(self):
        vhash1 = generate_vhash("TestClass", 1, {"value": 42}, {"subject": 1})
        vhash2 = generate_vhash("TestClass", 1, {"value": 42}, {"subject": 2})
        assert vhash1 != vhash2

    def test_vhash_with_numpy_data(self):
        arr = np.array([1.0, 2.0, 3.0])
        vhash = generate_vhash("ArrayClass", 1, arr, {"subject": 1})
        assert isinstance(vhash, str)
        assert len(vhash) == 16

    def test_vhash_with_numpy_deterministic(self):
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0, 3.0])
        vhash1 = generate_vhash("ArrayClass", 1, arr1, {"subject": 1})
        vhash2 = generate_vhash("ArrayClass", 1, arr2, {"subject": 1})
        assert vhash1 == vhash2

    def test_vhash_metadata_order_independent(self):
        """Metadata order should not affect vhash."""
        vhash1 = generate_vhash(
            "TestClass", 1, {"value": 42}, {"subject": 1, "trial": 2}
        )
        vhash2 = generate_vhash(
            "TestClass", 1, {"value": 42}, {"trial": 2, "subject": 1}
        )
        assert vhash1 == vhash2

    def test_vhash_complex_metadata(self):
        vhash = generate_vhash(
            "TestClass",
            1,
            {"value": 42},
            {
                "subject": 1,
                "trial": 2,
                "condition": "control",
                "date": "2024-01-01",
            },
        )
        assert isinstance(vhash, str)
        assert len(vhash) == 16
