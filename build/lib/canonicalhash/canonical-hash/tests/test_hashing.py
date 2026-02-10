"""Tests for canonical hashing."""

import pytest

from canonicalhash import canonical_hash, generate_record_id


class TestCanonicalHash:
    """Tests for the canonical_hash function."""

    def test_primitives(self):
        """Test hashing of primitive types."""
        # Same value should produce same hash
        assert canonical_hash(42) == canonical_hash(42)
        assert canonical_hash("hello") == canonical_hash("hello")
        assert canonical_hash(3.14) == canonical_hash(3.14)
        assert canonical_hash(True) == canonical_hash(True)
        assert canonical_hash(None) == canonical_hash(None)

        # Different values should produce different hashes
        assert canonical_hash(42) != canonical_hash(43)
        assert canonical_hash("hello") != canonical_hash("world")

    def test_hash_format(self):
        """Test that hash is a 16-character hex string."""
        h = canonical_hash(42)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_lists(self):
        """Test hashing of lists."""
        assert canonical_hash([1, 2, 3]) == canonical_hash([1, 2, 3])
        assert canonical_hash([1, 2, 3]) != canonical_hash([1, 2, 4])
        assert canonical_hash([1, 2, 3]) != canonical_hash([3, 2, 1])  # Order matters

    def test_tuples(self):
        """Test hashing of tuples."""
        assert canonical_hash((1, 2)) == canonical_hash((1, 2))
        assert canonical_hash((1, 2)) != canonical_hash([1, 2])  # Type matters

    def test_dicts(self):
        """Test hashing of dicts."""
        # Order should not matter for dicts
        assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})

        # Different values should produce different hashes
        assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})
        assert canonical_hash({"a": 1}) != canonical_hash({"b": 1})

    def test_nested_structures(self):
        """Test hashing of nested structures."""
        nested = {"key": [1, 2, {"inner": "value"}]}
        assert canonical_hash(nested) == canonical_hash(nested)

    def test_numpy_arrays(self):
        """Test hashing of numpy arrays."""
        pytest.importorskip("numpy")
        import numpy as np

        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])
        arr3 = np.array([1, 2, 4])

        assert canonical_hash(arr1) == canonical_hash(arr2)
        assert canonical_hash(arr1) != canonical_hash(arr3)

    def test_numpy_shape_matters(self):
        """Test that array shape affects hash."""
        pytest.importorskip("numpy")
        import numpy as np

        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([1, 2, 3, 4])

        assert canonical_hash(arr1) != canonical_hash(arr2)

    def test_pandas_dataframe(self):
        """Test hashing of pandas DataFrames."""
        pytest.importorskip("pandas")
        import pandas as pd

        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df3 = pd.DataFrame({"a": [1, 2], "b": [3, 5]})

        assert canonical_hash(df1) == canonical_hash(df2)
        assert canonical_hash(df1) != canonical_hash(df3)


class TestGenerateRecordId:
    """Tests for the generate_record_id function."""

    def test_basic_generation(self):
        """Test basic record ID generation."""
        rid = generate_record_id(
            class_name="MyClass",
            schema_version=1,
            content_hash="abc123",
            metadata={"subject": 1},
        )

        assert len(rid) == 16
        assert all(c in "0123456789abcdef" for c in rid)

    def test_deterministic(self):
        """Test that same inputs produce same ID."""
        args = {
            "class_name": "MyClass",
            "schema_version": 1,
            "content_hash": "abc123",
            "metadata": {"subject": 1, "trial": 2},
        }

        assert generate_record_id(**args) == generate_record_id(**args)

    def test_different_inputs(self):
        """Test that different inputs produce different IDs."""
        base = {
            "class_name": "MyClass",
            "schema_version": 1,
            "content_hash": "abc123",
            "metadata": {"subject": 1},
        }

        rid1 = generate_record_id(**base)
        rid2 = generate_record_id(**{**base, "class_name": "OtherClass"})
        rid3 = generate_record_id(**{**base, "schema_version": 2})
        rid4 = generate_record_id(**{**base, "content_hash": "xyz789"})
        rid5 = generate_record_id(**{**base, "metadata": {"subject": 2}})

        all_ids = [rid1, rid2, rid3, rid4, rid5]
        assert len(set(all_ids)) == 5  # All unique
