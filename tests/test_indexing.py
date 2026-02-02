"""Tests for index/loc/iloc functionality in scidb."""

import pytest
import numpy as np
import pandas as pd

from scidb import BaseVariable


class ListValue(BaseVariable):
    """Variable that stores a list/array with multiple rows."""

    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        return pd.DataFrame({"value": self.data})

    @classmethod
    def from_db(cls, df: pd.DataFrame):
        return df["value"].values


class TestSaveWithIndex:
    """Test saving data with an index."""

    def test_save_with_range_index(self, db):
        """Save with range() as index."""
        data = [10, 20, 30, 40, 50]
        record_id = ListValue.save(data, db=db, index=range(5), subject=1)

        assert record_id is not None

        # Load and verify
        loaded = ListValue.load(db=db, subject=1)
        np.testing.assert_array_equal(loaded.data, data)

    def test_save_with_list_index(self, db):
        """Save with list as index."""
        data = [100, 200, 300]
        index = ["a", "b", "c"]
        record_id = ListValue.save(data, db=db, index=index, subject=2)

        assert record_id is not None

        # Load and verify
        loaded = ListValue.load(db=db, subject=2)
        np.testing.assert_array_equal(loaded.data, data)

    def test_save_with_numeric_list_index(self, db):
        """Save with numeric list as index."""
        data = [1.5, 2.5, 3.5, 4.5]
        index = [0, 2, 4, 6]
        record_id = ListValue.save(data, db=db, index=index, subject=3)

        assert record_id is not None

    def test_save_index_length_mismatch_raises(self, db):
        """Index length not matching data should raise ValueError."""
        data = [1, 2, 3, 4, 5]
        index = [0, 1, 2]  # Wrong length

        with pytest.raises(ValueError, match="Index length"):
            ListValue.save(data, db=db, index=index, subject=4)

    def test_save_without_index(self, db):
        """Save without index should work as before."""
        data = [1, 2, 3]
        record_id = ListValue.save(data, db=db, subject=5)

        assert record_id is not None

        loaded = ListValue.load(db=db, subject=5)
        np.testing.assert_array_equal(loaded.data, data)


class TestLoadWithLoc:
    """Test loading data with loc (label-based indexing)."""

    def test_load_with_loc_single_value(self, db):
        """Load single element by label."""
        data = [10, 20, 30, 40, 50]
        index = ["a", "b", "c", "d", "e"]
        ListValue.save(data, db=db, index=index, subject=1)

        loaded = ListValue.load(db=db, subject=1, loc="c")
        assert len(loaded.data) == 1
        assert loaded.data[0] == 30

    def test_load_with_loc_list(self, db):
        """Load multiple elements by label list."""
        data = [10, 20, 30, 40, 50]
        index = ["a", "b", "c", "d", "e"]
        ListValue.save(data, db=db, index=index, subject=2)

        loaded = ListValue.load(db=db, subject=2, loc=["a", "c", "e"])
        np.testing.assert_array_equal(loaded.data, [10, 30, 50])

    def test_load_with_loc_range(self, db):
        """Load elements using range as loc."""
        data = [100, 200, 300, 400, 500]
        index = range(5)
        ListValue.save(data, db=db, index=index, subject=3)

        loaded = ListValue.load(db=db, subject=3, loc=range(1, 4))
        np.testing.assert_array_equal(loaded.data, [200, 300, 400])

    def test_load_with_loc_slice(self, db):
        """Load elements using slice as loc."""
        data = [10, 20, 30, 40, 50]
        index = ["a", "b", "c", "d", "e"]
        ListValue.save(data, db=db, index=index, subject=4)

        loaded = ListValue.load(db=db, subject=4, loc=slice("b", "d"))
        np.testing.assert_array_equal(loaded.data, [20, 30, 40])

    def test_load_with_loc_numeric_index(self, db):
        """Load with loc using numeric index."""
        data = [1.0, 2.0, 3.0, 4.0]
        index = [0, 10, 20, 30]
        ListValue.save(data, db=db, index=index, subject=5)

        loaded = ListValue.load(db=db, subject=5, loc=10)
        assert len(loaded.data) == 1
        assert loaded.data[0] == 2.0

    def test_load_without_loc_returns_all(self, db):
        """Load without loc should return all data."""
        data = [1, 2, 3, 4, 5]
        index = range(5)
        ListValue.save(data, db=db, index=index, subject=6)

        loaded = ListValue.load(db=db, subject=6)
        np.testing.assert_array_equal(loaded.data, data)


class TestLoadWithIloc:
    """Test loading data with iloc (integer position-based indexing)."""

    def test_load_with_iloc_single_value(self, db):
        """Load single element by position."""
        data = [10, 20, 30, 40, 50]
        index = ["a", "b", "c", "d", "e"]
        ListValue.save(data, db=db, index=index, subject=1)

        loaded = ListValue.load(db=db, subject=1, iloc=2)
        assert len(loaded.data) == 1
        assert loaded.data[0] == 30

    def test_load_with_iloc_list(self, db):
        """Load multiple elements by position list."""
        data = [10, 20, 30, 40, 50]
        index = ["a", "b", "c", "d", "e"]
        ListValue.save(data, db=db, index=index, subject=2)

        loaded = ListValue.load(db=db, subject=2, iloc=[0, 2, 4])
        np.testing.assert_array_equal(loaded.data, [10, 30, 50])

    def test_load_with_iloc_range(self, db):
        """Load elements using range as iloc."""
        data = [100, 200, 300, 400, 500]
        ListValue.save(data, db=db, index=range(5), subject=3)

        loaded = ListValue.load(db=db, subject=3, iloc=range(1, 4))
        np.testing.assert_array_equal(loaded.data, [200, 300, 400])

    def test_load_with_iloc_slice(self, db):
        """Load elements using slice as iloc."""
        data = [10, 20, 30, 40, 50]
        ListValue.save(data, db=db, index=range(5), subject=4)

        loaded = ListValue.load(db=db, subject=4, iloc=slice(1, 4))
        np.testing.assert_array_equal(loaded.data, [20, 30, 40])

    def test_load_with_iloc_negative_index(self, db):
        """Load with iloc using negative index."""
        data = [1, 2, 3, 4, 5]
        ListValue.save(data, db=db, index=range(5), subject=5)

        loaded = ListValue.load(db=db, subject=5, iloc=-1)
        assert len(loaded.data) == 1
        assert loaded.data[0] == 5


class TestLocIlocValidation:
    """Test validation of loc/iloc parameters."""

    def test_cannot_use_both_loc_and_iloc(self, db):
        """Using both loc and iloc should raise ValueError."""
        data = [1, 2, 3]
        ListValue.save(data, db=db, index=range(3), subject=1)

        with pytest.raises(ValueError, match="Cannot specify both"):
            ListValue.load(db=db, subject=1, loc=0, iloc=0)


class TestIndexPreservedInParquet:
    """Test that index is preserved through save/load cycle."""

    def test_index_preserved_on_load(self, db):
        """Index should be accessible after loading."""
        data = [10, 20, 30]
        index = ["x", "y", "z"]
        ListValue.save(data, db=db, index=index, subject=1)

        # Load with loc to verify index is preserved
        loaded = ListValue.load(db=db, subject=1, loc="y")
        assert len(loaded.data) == 1
        assert loaded.data[0] == 20

    def test_numeric_index_preserved(self, db):
        """Numeric index should be preserved."""
        data = [100, 200, 300, 400]
        index = [5, 10, 15, 20]
        ListValue.save(data, db=db, index=index, subject=2)

        # Load specific index
        loaded = ListValue.load(db=db, subject=2, loc=15)
        assert len(loaded.data) == 1
        assert loaded.data[0] == 300


class TestReservedKeys:
    """Test that index, loc, iloc are reserved keys."""

    def test_index_is_reserved(self):
        """'index' should be in reserved keys."""
        assert "index" in BaseVariable._reserved_keys

    def test_loc_is_reserved(self):
        """'loc' should be in reserved keys."""
        assert "loc" in BaseVariable._reserved_keys

    def test_iloc_is_reserved(self):
        """'iloc' should be in reserved keys."""
        assert "iloc" in BaseVariable._reserved_keys
