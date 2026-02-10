"""Tests for thunk hashing utilities."""

import pytest

from thunk import canonical_hash


class TestCanonicalHash:
    """Test canonical_hash function."""

    def test_primitives(self):
        """Primitives should produce deterministic hashes."""
        assert canonical_hash(42) == canonical_hash(42)
        assert canonical_hash("hello") == canonical_hash("hello")
        assert canonical_hash(3.14) == canonical_hash(3.14)
        assert canonical_hash(True) == canonical_hash(True)
        assert canonical_hash(None) == canonical_hash(None)

    def test_different_values_different_hashes(self):
        """Different values should produce different hashes."""
        assert canonical_hash(1) != canonical_hash(2)
        assert canonical_hash("a") != canonical_hash("b")
        assert canonical_hash(True) != canonical_hash(False)

    def test_lists(self):
        """Lists should hash deterministically."""
        assert canonical_hash([1, 2, 3]) == canonical_hash([1, 2, 3])
        assert canonical_hash([1, 2, 3]) != canonical_hash([1, 2, 4])
        assert canonical_hash([1, 2, 3]) != canonical_hash([3, 2, 1])  # Order matters

    def test_tuples(self):
        """Tuples should hash deterministically."""
        assert canonical_hash((1, 2, 3)) == canonical_hash((1, 2, 3))
        assert canonical_hash((1, 2, 3)) != canonical_hash([1, 2, 3])  # Type matters

    def test_dicts(self):
        """Dicts should hash deterministically regardless of key order."""
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}
        assert canonical_hash(d1) == canonical_hash(d2)

        d3 = {"a": 1, "b": 3}
        assert canonical_hash(d1) != canonical_hash(d3)

    def test_nested_structures(self):
        """Nested structures should hash correctly."""
        nested = {"list": [1, 2, 3], "dict": {"x": 1}, "tuple": (4, 5)}
        assert canonical_hash(nested) == canonical_hash(nested.copy())

    def test_hash_length(self):
        """Hash should be 16 characters (64 bits)."""
        h = canonical_hash(42)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


class TestCanonicalHashWithNumpy:
    """Test canonical_hash with numpy arrays."""

    @pytest.fixture
    def np(self):
        numpy = pytest.importorskip("numpy")
        return numpy

    def test_numpy_array(self, np):
        """Numpy arrays should hash based on content."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0, 3.0])
        assert canonical_hash(arr1) == canonical_hash(arr2)

    def test_numpy_array_different_values(self, np):
        """Different array values should produce different hashes."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0, 4.0])
        assert canonical_hash(arr1) != canonical_hash(arr2)

    def test_numpy_array_different_dtype(self, np):
        """Different dtypes should produce different hashes."""
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        arr2 = np.array([1, 2, 3], dtype=np.int64)
        assert canonical_hash(arr1) != canonical_hash(arr2)

    def test_numpy_array_different_shape(self, np):
        """Different shapes should produce different hashes."""
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([1, 2, 3, 4])
        assert canonical_hash(arr1) != canonical_hash(arr2)


class TestCanonicalHashWithPandas:
    """Test canonical_hash with pandas objects."""

    @pytest.fixture
    def pd(self):
        pandas = pytest.importorskip("pandas")
        return pandas

    def test_dataframe(self, pd):
        """DataFrames should hash based on content."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert canonical_hash(df1) == canonical_hash(df2)

    def test_dataframe_different_values(self, pd):
        """Different DataFrame values should produce different hashes."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [1, 2], "b": [3, 5]})
        assert canonical_hash(df1) != canonical_hash(df2)

    def test_series(self, pd):
        """Series should hash based on content."""
        s1 = pd.Series([1, 2, 3], name="x")
        s2 = pd.Series([1, 2, 3], name="x")
        assert canonical_hash(s1) == canonical_hash(s2)
