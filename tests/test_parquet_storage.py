"""Tests for Parquet file storage with metadata-based folder hierarchy."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scidb.parquet_storage import (
    compute_folder_path,
    compute_parquet_path,
    delete_parquet,
    extract_metadata_from_path,
    extract_table_name_from_path,
    extract_record_id_from_filename,
    get_parquet_root,
    list_parquet_files,
    parse_metadata_order,
    read_parquet,
    write_parquet,
    _sanitize_path_component,
)


class TestGetParquetRoot:
    """Tests for get_parquet_root."""

    def test_creates_parquet_folder_path(self, tmp_path):
        """Test that parquet root is db_folder/parquet/."""
        db_path = tmp_path / "data" / "experiment.db"
        root = get_parquet_root(db_path)

        assert root == tmp_path / "data" / "parquet"

    def test_with_db_in_current_dir(self, tmp_path):
        """Test with database in current directory."""
        db_path = tmp_path / "test.db"
        root = get_parquet_root(db_path)

        assert root == tmp_path / "parquet"


class TestComputeFolderPath:
    """Tests for compute_folder_path."""

    def test_alphabetical_order_by_default(self):
        """Test that keys are sorted alphabetically by default."""
        metadata = {"subject": 1, "visit": 2, "arm": "left"}
        path = compute_folder_path(metadata)

        # arm < subject < visit alphabetically
        assert path == Path("arm/left/subject/1/visit/2")

    def test_custom_order(self):
        """Test custom metadata order."""
        metadata = {"subject": 1, "visit": 2, "arm": "left"}
        path = compute_folder_path(metadata, metadata_order=["subject", "visit", "arm"])

        assert path == Path("subject/1/visit/2/arm/left")

    def test_partial_order(self):
        """Test with order containing subset of keys."""
        metadata = {"subject": 1, "visit": 2, "arm": "left"}
        path = compute_folder_path(metadata, metadata_order=["subject"])

        # Only subject is included
        assert path == Path("subject/1")

    def test_empty_metadata(self):
        """Test with empty metadata."""
        path = compute_folder_path({})
        assert path == Path(".")

    def test_invalid_order_key(self):
        """Test error when order contains non-existent key."""
        metadata = {"subject": 1}
        with pytest.raises(ValueError, match="metadata_order contains keys not in metadata"):
            compute_folder_path(metadata, metadata_order=["subject", "nonexistent"])

    def test_numeric_values(self):
        """Test that numeric values are converted to strings."""
        metadata = {"subject": 1, "trial": 2.5}
        path = compute_folder_path(metadata)

        assert path == Path("subject/1/trial/2.5")

    def test_string_values(self):
        """Test string values."""
        metadata = {"condition": "control", "phase": "training"}
        path = compute_folder_path(metadata)

        assert path == Path("condition/control/phase/training")


class TestSanitizePathComponent:
    """Tests for _sanitize_path_component."""

    def test_removes_slashes(self):
        """Test that slashes are replaced."""
        assert _sanitize_path_component("a/b") == "a_b"
        assert _sanitize_path_component("a\\b") == "a_b"

    def test_removes_special_chars(self):
        """Test that special characters are replaced."""
        assert _sanitize_path_component("a:b*c?d") == "a_b_c_d"

    def test_strips_dots_and_spaces(self):
        """Test that leading/trailing dots and spaces are stripped."""
        assert _sanitize_path_component("...test...") == "test"
        assert _sanitize_path_component("  test  ") == "test"

    def test_empty_becomes_underscore(self):
        """Test that empty string becomes underscore."""
        assert _sanitize_path_component("") == "_"
        assert _sanitize_path_component("...") == "_"

    def test_normal_strings_unchanged(self):
        """Test that normal strings pass through."""
        assert _sanitize_path_component("hello_world") == "hello_world"
        assert _sanitize_path_component("test123") == "test123"


class TestParseMetadataOrder:
    """Tests for parse_metadata_order."""

    def test_with_leading_slash(self):
        """Test parsing with leading slash."""
        assert parse_metadata_order("/visit/subject") == ["visit", "subject"]

    def test_without_leading_slash(self):
        """Test parsing without leading slash."""
        assert parse_metadata_order("visit/subject") == ["visit", "subject"]

    def test_with_trailing_slash(self):
        """Test parsing with trailing slash."""
        assert parse_metadata_order("visit/subject/") == ["visit", "subject"]

    def test_single_key(self):
        """Test parsing single key."""
        assert parse_metadata_order("/subject") == ["subject"]

    def test_empty_string(self):
        """Test parsing empty string."""
        assert parse_metadata_order("") == []
        assert parse_metadata_order("/") == []


class TestComputeParquetPath:
    """Tests for compute_parquet_path."""

    def test_full_path_generation(self):
        """Test complete path generation."""
        path = compute_parquet_path(
            table_name="sensor_reading",
            record_id="abc123def456",
            metadata={"subject": 1, "visit": 2},
            parquet_root=Path("/data/parquet"),
        )

        assert path == Path("/data/parquet/sensor_reading/subject/1/visit/2/abc123def456.parquet")

    def test_with_custom_order(self):
        """Test with custom metadata order."""
        path = compute_parquet_path(
            table_name="sensor_reading",
            record_id="abc123",
            metadata={"subject": 1, "visit": 2},
            parquet_root=Path("/data/parquet"),
            metadata_order=["visit", "subject"],
        )

        assert path == Path("/data/parquet/sensor_reading/visit/2/subject/1/abc123.parquet")

    def test_empty_metadata(self):
        """Test with no metadata (flat storage within table folder)."""
        path = compute_parquet_path(
            table_name="sensor_reading",
            record_id="abc123",
            metadata={},
            parquet_root=Path("/data/parquet"),
        )

        assert path == Path("/data/parquet/sensor_reading/abc123.parquet")


class TestWriteAndReadParquet:
    """Tests for write_parquet and read_parquet."""

    def test_write_creates_directories(self, tmp_path):
        """Test that write_parquet creates parent directories."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        path = tmp_path / "a" / "b" / "c" / "test.parquet"

        write_parquet(df, path)

        assert path.exists()
        assert path.parent.exists()

    def test_roundtrip(self, tmp_path):
        """Test write and read roundtrip."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
        })
        path = tmp_path / "test.parquet"

        write_parquet(df, path)
        result = read_parquet(path)

        pd.testing.assert_frame_equal(df, result)

    def test_read_nonexistent_raises(self, tmp_path):
        """Test that reading nonexistent file raises error."""
        path = tmp_path / "nonexistent.parquet"

        with pytest.raises(FileNotFoundError):
            read_parquet(path)

    def test_compression(self, tmp_path):
        """Test different compression options."""
        df = pd.DataFrame({"x": range(1000)})

        # Write with different compressions
        for compression in ["snappy", "gzip", "none"]:
            path = tmp_path / f"test_{compression}.parquet"
            write_parquet(df, path, compression=compression)
            result = read_parquet(path)
            pd.testing.assert_frame_equal(df, result)


class TestDeleteParquet:
    """Tests for delete_parquet."""

    def test_deletes_file(self, tmp_path):
        """Test that file is deleted."""
        path = tmp_path / "test.parquet"
        path.touch()

        result = delete_parquet(path)

        assert result is True
        assert not path.exists()

    def test_returns_false_if_not_exists(self, tmp_path):
        """Test return value when file doesn't exist."""
        path = tmp_path / "nonexistent.parquet"

        result = delete_parquet(path)

        assert result is False

    def test_cleans_up_empty_dirs(self, tmp_path):
        """Test that empty parent directories are removed."""
        # Create nested structure
        parquet_root = tmp_path / "parquet"
        path = parquet_root / "subject" / "1" / "visit" / "2" / "test.parquet"
        path.parent.mkdir(parents=True)
        path.touch()

        delete_parquet(path, cleanup_empty_dirs=True)

        # All empty directories should be removed up to parquet/
        assert not (parquet_root / "subject" / "1" / "visit" / "2").exists()
        assert not (parquet_root / "subject" / "1" / "visit").exists()
        assert not (parquet_root / "subject" / "1").exists()
        assert not (parquet_root / "subject").exists()
        # parquet/ itself remains (it's the root)
        assert parquet_root.exists()

    def test_preserves_nonempty_dirs(self, tmp_path):
        """Test that non-empty directories are preserved."""
        parquet_root = tmp_path / "parquet"
        path1 = parquet_root / "subject" / "1" / "test1.parquet"
        path2 = parquet_root / "subject" / "1" / "test2.parquet"
        path1.parent.mkdir(parents=True)
        path1.touch()
        path2.touch()

        delete_parquet(path1, cleanup_empty_dirs=True)

        # Directory should remain because path2 still exists
        assert path1.parent.exists()
        assert path2.exists()


class TestListParquetFiles:
    """Tests for list_parquet_files."""

    @pytest.fixture
    def populated_parquet_root(self, tmp_path):
        """Create a parquet root with test files.

        Structure:
            parquet/
                sensor_reading/
                    subject/1/visit/1/abc123.parquet
                    subject/1/visit/2/def456.parquet
                    subject/2/visit/1/ghi789.parquet
                processed_signal/
                    subject/1/visit/1/jkl012.parquet
        """
        root = tmp_path / "parquet"

        # Create some test files (table_name is first folder)
        files = [
            "sensor_reading/subject/1/visit/1/abc123.parquet",
            "sensor_reading/subject/1/visit/2/def456.parquet",
            "sensor_reading/subject/2/visit/1/ghi789.parquet",
            "processed_signal/subject/1/visit/1/jkl012.parquet",
        ]

        for f in files:
            path = root / f
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write minimal valid parquet
            pd.DataFrame({"x": [1]}).to_parquet(path)

        return root

    def test_list_all(self, populated_parquet_root):
        """Test listing all files."""
        files = list_parquet_files(populated_parquet_root)

        assert len(files) == 4

    def test_filter_by_table_name(self, populated_parquet_root):
        """Test filtering by table name (first folder)."""
        files = list_parquet_files(populated_parquet_root, table_name="sensor_reading")

        assert len(files) == 3
        for f in files:
            assert "sensor_reading" in str(f)

    def test_filter_by_metadata(self, populated_parquet_root):
        """Test filtering by metadata."""
        files = list_parquet_files(
            populated_parquet_root,
            metadata_filter={"subject": 1}
        )

        assert len(files) == 3
        for f in files:
            assert "subject/1" in str(f)

    def test_combined_filters(self, populated_parquet_root):
        """Test combining table name and metadata filters."""
        files = list_parquet_files(
            populated_parquet_root,
            table_name="sensor_reading",
            metadata_filter={"subject": 1, "visit": 1}
        )

        assert len(files) == 1
        assert "abc123.parquet" in str(files[0])

    def test_empty_root(self, tmp_path):
        """Test with nonexistent root."""
        root = tmp_path / "nonexistent"
        files = list_parquet_files(root)

        assert files == []

    def test_nonexistent_table(self, populated_parquet_root):
        """Test filtering by non-existent table name."""
        files = list_parquet_files(populated_parquet_root, table_name="nonexistent")

        assert files == []


class TestExtractMetadataFromPath:
    """Tests for extract_metadata_from_path."""

    def test_extracts_metadata(self, tmp_path):
        """Test metadata extraction from path (skips table_name folder)."""
        root = tmp_path / "parquet"
        # Structure: parquet_root / table_name / key / value / ... / record_id.parquet
        path = root / "sensor_reading" / "subject" / "1" / "visit" / "2" / "abc123.parquet"

        metadata = extract_metadata_from_path(path, root)

        assert metadata == {"subject": "1", "visit": "2"}

    def test_nested_metadata(self, tmp_path):
        """Test with deeper nesting."""
        root = tmp_path / "parquet"
        path = root / "sensor_reading" / "experiment" / "exp1" / "subject" / "5" / "trial" / "3" / "abc.parquet"

        metadata = extract_metadata_from_path(path, root)

        assert metadata == {
            "experiment": "exp1",
            "subject": "5",
            "trial": "3",
        }

    def test_flat_file_in_table_folder(self, tmp_path):
        """Test with file directly in table folder (no metadata)."""
        root = tmp_path / "parquet"
        path = root / "sensor_reading" / "abc123.parquet"

        metadata = extract_metadata_from_path(path, root)

        assert metadata == {}

    def test_path_not_under_root(self, tmp_path):
        """Test with path not under parquet root."""
        root = tmp_path / "parquet"
        path = tmp_path / "other" / "test.parquet"

        metadata = extract_metadata_from_path(path, root)

        assert metadata == {}


class TestExtractTableNameFromPath:
    """Tests for extract_table_name_from_path."""

    def test_extracts_table_name(self, tmp_path):
        """Test table name extraction (first folder after root)."""
        root = tmp_path / "parquet"
        path = root / "sensor_reading" / "subject" / "1" / "abc123.parquet"

        table_name = extract_table_name_from_path(path, root)

        assert table_name == "sensor_reading"

    def test_flat_file(self, tmp_path):
        """Test with file directly in table folder."""
        root = tmp_path / "parquet"
        path = root / "processed_signal" / "def456.parquet"

        table_name = extract_table_name_from_path(path, root)

        assert table_name == "processed_signal"

    def test_path_not_under_root(self, tmp_path):
        """Test with path not under parquet root."""
        root = tmp_path / "parquet"
        path = tmp_path / "other" / "test.parquet"

        table_name = extract_table_name_from_path(path, root)

        assert table_name is None


class TestExtractVhashFromFilename:
    """Tests for extract_record_id_from_filename."""

    def test_extract_record_id(self):
        """Test record_id extraction (filename is just record_id.parquet)."""
        assert extract_record_id_from_filename("abc123def456.parquet") == "abc123def456"
        assert extract_record_id_from_filename("xyz.parquet") == "xyz"

    def test_not_parquet_extension(self):
        """Test with non-parquet extension."""
        assert extract_record_id_from_filename("abc123.csv") is None
        assert extract_record_id_from_filename("abc123") is None

    def test_empty_filename(self):
        """Test with just the extension."""
        assert extract_record_id_from_filename(".parquet") is None
