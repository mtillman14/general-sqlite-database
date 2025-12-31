"""Tests for scidb.storage module."""

import pytest
import numpy as np
import pandas as pd

from scidb.storage import (
    serialize_dataframe,
    deserialize_dataframe,
    adapt_dataframe,
    convert_dataframe,
    register_adapters,
)


class TestDataFrameSerialization:
    """Test DataFrame serialization and deserialization."""

    def test_serialize_simple_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        blob = serialize_dataframe(df)
        assert isinstance(blob, bytes)
        assert len(blob) > 0

    def test_deserialize_simple_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        blob = serialize_dataframe(df)
        restored = deserialize_dataframe(blob)
        pd.testing.assert_frame_equal(df, restored)

    def test_roundtrip_empty_dataframe(self):
        df = pd.DataFrame()
        blob = serialize_dataframe(df)
        restored = deserialize_dataframe(blob)
        pd.testing.assert_frame_equal(df, restored)

    def test_roundtrip_single_column(self):
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        blob = serialize_dataframe(df)
        restored = deserialize_dataframe(blob)
        pd.testing.assert_frame_equal(df, restored)

    def test_roundtrip_single_row(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        blob = serialize_dataframe(df)
        restored = deserialize_dataframe(blob)
        pd.testing.assert_frame_equal(df, restored)

    def test_roundtrip_mixed_types(self):
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        })
        blob = serialize_dataframe(df)
        restored = deserialize_dataframe(blob)
        pd.testing.assert_frame_equal(df, restored)

    def test_roundtrip_with_nulls(self):
        df = pd.DataFrame({
            "a": [1, None, 3],
            "b": [None, 2.0, None],
        })
        blob = serialize_dataframe(df)
        restored = deserialize_dataframe(blob)
        pd.testing.assert_frame_equal(df, restored)

    def test_roundtrip_with_index(self):
        df = pd.DataFrame(
            {"a": [1, 2, 3]},
            index=["x", "y", "z"]
        )
        blob = serialize_dataframe(df)
        restored = deserialize_dataframe(blob)
        pd.testing.assert_frame_equal(df, restored)

    def test_roundtrip_multiindex(self):
        df = pd.DataFrame(
            {"value": [1, 2, 3, 4]},
            index=pd.MultiIndex.from_tuples([
                ("a", 1), ("a", 2), ("b", 1), ("b", 2)
            ])
        )
        blob = serialize_dataframe(df)
        restored = deserialize_dataframe(blob)
        pd.testing.assert_frame_equal(df, restored)

    def test_roundtrip_datetime_column(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=3),
            "value": [1, 2, 3],
        })
        blob = serialize_dataframe(df)
        restored = deserialize_dataframe(blob)
        pd.testing.assert_frame_equal(df, restored)

    def test_roundtrip_categorical(self):
        df = pd.DataFrame({
            "category": pd.Categorical(["a", "b", "a", "c"]),
        })
        blob = serialize_dataframe(df)
        restored = deserialize_dataframe(blob)
        pd.testing.assert_frame_equal(df, restored)

    def test_roundtrip_large_dataframe(self):
        df = pd.DataFrame({
            "a": np.random.rand(1000),
            "b": np.random.rand(1000),
            "c": np.random.rand(1000),
        })
        blob = serialize_dataframe(df)
        restored = deserialize_dataframe(blob)
        pd.testing.assert_frame_equal(df, restored)

    def test_roundtrip_with_bytes_column(self):
        """Test that DataFrames containing bytes can be serialized."""
        df = pd.DataFrame({
            "data": [b"hello", b"world"],
            "id": [1, 2],
        })
        blob = serialize_dataframe(df)
        restored = deserialize_dataframe(blob)
        pd.testing.assert_frame_equal(df, restored)


class TestSQLiteAdapters:
    """Test SQLite adapter functions."""

    def test_adapt_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        blob = adapt_dataframe(df)
        assert isinstance(blob, bytes)

    def test_convert_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        blob = adapt_dataframe(df)
        restored = convert_dataframe(blob)
        pd.testing.assert_frame_equal(df, restored)

    def test_adapt_equals_serialize(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert adapt_dataframe(df) == serialize_dataframe(df)

    def test_register_adapters_runs_without_error(self):
        """Test that register_adapters() can be called without error."""
        # Should not raise
        register_adapters()
        # Can be called multiple times
        register_adapters()
