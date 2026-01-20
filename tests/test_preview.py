"""Tests for preview generation and export functionality."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from scidb.preview import generate_preview
from scidb.variable import BaseVariable


class TestGeneratePreview:
    """Test the preview generation function."""

    def test_empty_dataframe(self):
        """Empty DataFrame should return '(empty)'."""
        df = pd.DataFrame()
        assert generate_preview(df) == "(empty)"

    def test_single_row_scalar(self):
        """Single row DataFrame shows values directly."""
        df = pd.DataFrame({"value": [42]})
        preview = generate_preview(df)
        assert "[1 rows x 1 cols]" in preview
        assert "value=42" in preview

    def test_single_row_multiple_columns(self):
        """Single row with multiple columns shows all values."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        preview = generate_preview(df)
        assert "a=1" in preview
        assert "b=2" in preview
        assert "c=3" in preview

    def test_multi_row_shows_stats(self):
        """Multi-row DataFrame shows statistics."""
        df = pd.DataFrame({
            "index": range(100),
            "value": np.linspace(0, 10, 100)
        })
        preview = generate_preview(df)
        assert "[100 rows x 2 cols]" in preview
        assert "min=" in preview
        assert "max=" in preview
        assert "mean=" in preview

    def test_multi_row_shows_sample(self):
        """Multi-row DataFrame shows sample values."""
        df = pd.DataFrame({
            "index": range(10),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        })
        preview = generate_preview(df)
        assert "Sample:" in preview
        assert "1" in preview  # First value should appear

    def test_many_columns_truncated(self):
        """Many columns are truncated in preview."""
        df = pd.DataFrame({f"col_{i}": [i] for i in range(10)})
        preview = generate_preview(df)
        assert "..." in preview

    def test_float_formatting(self):
        """Floats are formatted reasonably."""
        df = pd.DataFrame({"value": [0.123456789]})
        preview = generate_preview(df)
        # Should not show all decimal places
        assert "0.123456789" not in preview
        assert "0.1235" in preview or "0.1234" in preview

    def test_scientific_notation_for_large_numbers(self):
        """Large numbers use scientific notation."""
        df = pd.DataFrame({"value": [1e10]})
        preview = generate_preview(df)
        assert "e" in preview.lower() or "E" in preview

    def test_bytes_data(self):
        """Bytes data shows length."""
        df = pd.DataFrame({"data": [b"hello world"]})
        preview = generate_preview(df)
        assert "bytes" in preview

    def test_max_length_truncation(self):
        """Preview is truncated to max_length."""
        df = pd.DataFrame({"long_column_name_" + str(i): [i] for i in range(20)})
        preview = generate_preview(df, max_length=100)
        assert len(preview) <= 100
        assert preview.endswith("...")

    def test_nan_values(self):
        """NaN values are handled."""
        df = pd.DataFrame({"value": [1.0, np.nan, 3.0]})
        preview = generate_preview(df)
        # Should not crash
        assert "[3 rows" in preview


class TestPreviewInDatabase:
    """Test that preview is stored in database."""

    def test_preview_column_populated(self, db, scalar_class):
        """Preview column should be populated on save."""
        scalar_class.save(42, db=db, key="test")

        # Check directly in database
        cursor = db.connection.execute(
            f"SELECT preview FROM {scalar_class.table_name()} WHERE 1=1"
        )
        row = cursor.fetchone()
        assert row is not None
        assert row["preview"] is not None
        assert "42" in row["preview"]

    def test_preview_column_for_array(self, db, array_class):
        """Preview shows stats for arrays."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        array_class.save(data, db=db, key="test")

        cursor = db.connection.execute(
            f"SELECT preview FROM {array_class.table_name()} WHERE 1=1"
        )
        row = cursor.fetchone()
        preview = row["preview"]
        assert preview is not None
        assert "[5 rows" in preview or "5" in preview


class TestPreviewDataMethod:
    """Test DatabaseManager.preview_data() method."""

    def test_preview_data_single_record(self, db, scalar_class):
        """preview_data returns formatted string for single record."""
        scalar_class.save(42, db=db, key="test", experiment="exp1")

        preview = db.preview_data(scalar_class, experiment="exp1")
        assert "ScalarValue" in preview
        assert "1 records" in preview
        assert "key=test" in preview
        assert "experiment=exp1" in preview

    def test_preview_data_multiple_records(self, db, scalar_class):
        """preview_data shows all matching records."""
        scalar_class.save(10, db=db, key="a", group="test")
        scalar_class.save(20, db=db, key="b", group="test")
        scalar_class.save(30, db=db, key="c", group="test")

        preview = db.preview_data(scalar_class, group="test")
        assert "3 records" in preview
        assert "key=a" in preview
        assert "key=b" in preview
        assert "key=c" in preview

    def test_preview_data_not_found(self, db, scalar_class):
        """preview_data raises NotFoundError when no matches."""
        from scidb.exceptions import NotFoundError

        scalar_class.save(42, db=db, existing="data")

        with pytest.raises(NotFoundError):
            db.preview_data(scalar_class, nonexistent="value")


class TestExportToCsv:
    """Test DatabaseManager.export_to_csv() method."""

    def test_export_single_record(self, db, scalar_class):
        """Export single record to CSV."""
        scalar_class.save(42, db=db, key="test")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name

        try:
            count = db.export_to_csv(scalar_class, path, key="test")
            assert count == 1

            # Read back and verify
            df = pd.read_csv(path)
            assert "value" in df.columns
            assert df["value"].iloc[0] == 42
            assert "_record_id" in df.columns
            assert "_meta_key" in df.columns
            assert df["_meta_key"].iloc[0] == "test"
        finally:
            os.unlink(path)

    def test_export_multiple_records(self, db, scalar_class):
        """Export multiple records to CSV."""
        scalar_class.save(10, db=db, item=1, group="export_test")
        scalar_class.save(20, db=db, item=2, group="export_test")
        scalar_class.save(30, db=db, item=3, group="export_test")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name

        try:
            count = db.export_to_csv(scalar_class, path, group="export_test")
            assert count == 3

            df = pd.read_csv(path)
            assert len(df) == 3
            assert set(df["value"]) == {10, 20, 30}
        finally:
            os.unlink(path)

    def test_export_array_data(self, db, array_class):
        """Export array data (multiple rows per record)."""
        data = np.array([1.0, 2.0, 3.0])
        array_class.save(data, db=db, key="array_test")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name

        try:
            count = db.export_to_csv(array_class, path, key="array_test")
            assert count == 1

            df = pd.read_csv(path)
            # Array has multiple rows in to_db() representation
            assert len(df) >= 1
        finally:
            os.unlink(path)


class TestVariableToCsv:
    """Test BaseVariable.to_csv() method."""

    def test_to_csv_scalar(self, db, scalar_class):
        """Export scalar variable to CSV."""
        scalar_class.save(42, db=db, key="test")
        var = scalar_class.load(db=db, key="test")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name

        try:
            var.to_csv(path)

            df = pd.read_csv(path)
            assert "value" in df.columns
            assert df["value"].iloc[0] == 42
        finally:
            os.unlink(path)

    def test_to_csv_array(self, db, array_class):
        """Export array variable to CSV."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        array_class.save(data, db=db, key="test")
        var = array_class.load(db=db, key="test")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name

        try:
            var.to_csv(path)

            df = pd.read_csv(path)
            # Check file was created and has content
            assert len(df) >= 1
        finally:
            os.unlink(path)


class TestVariableGetPreview:
    """Test BaseVariable.get_preview() method."""

    def test_get_preview_scalar(self, scalar_class):
        """get_preview works on scalar."""
        var = scalar_class(42)
        preview = var.get_preview()
        assert "42" in preview

    def test_get_preview_array(self, array_class):
        """get_preview works on array."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        var = array_class(data)
        preview = var.get_preview()
        assert "[5 rows" in preview or "5" in preview
