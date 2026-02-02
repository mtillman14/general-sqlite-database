"""Parquet file storage with metadata-based folder hierarchy."""

import os
from pathlib import Path

import pandas as pd


def get_parquet_root(db_path: Path) -> Path:
    """
    Get the root folder for Parquet files.

    Creates a 'parquet/' folder in the same directory as the database file.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        Path to the parquet root folder
    """
    return db_path.parent / "parquet"


def compute_folder_path(
    metadata: dict,
    metadata_order: list[str] | None = None,
) -> Path:
    """
    Compute the folder hierarchy path based on metadata.

    Creates a folder structure with key/value pairs, e.g.:
        {"subject": 1, "visit": 2} → subject/1/visit/2/

    Args:
        metadata: Dict of metadata key-value pairs
        metadata_order: Optional list of keys specifying folder order.
                       If not provided, keys are sorted alphabetically.

    Returns:
        Relative path for the folder hierarchy

    Raises:
        ValueError: If metadata_order contains keys not in metadata

    Example:
        >>> compute_folder_path({"subject": 1, "visit": 2})
        PosixPath('subject/1/visit/2')

        >>> compute_folder_path({"subject": 1, "visit": 2}, metadata_order=["visit", "subject"])
        PosixPath('visit/2/subject/1')
    """
    if not metadata:
        return Path(".")

    if metadata_order is not None:
        # Validate that all ordered keys exist in metadata
        missing = set(metadata_order) - set(metadata.keys())
        if missing:
            raise ValueError(
                f"metadata_order contains keys not in metadata: {missing}"
            )
        ordered_keys = metadata_order
    else:
        # Default: alphabetically sorted keys
        ordered_keys = sorted(metadata.keys())

    # Build path: key/value/key/value/...
    parts = []
    for key in ordered_keys:
        value = metadata[key]
        parts.append(str(key))
        parts.append(_sanitize_path_component(str(value)))

    return Path(*parts) if parts else Path(".")


def compute_parquet_path(
    table_name: str,
    record_id: str,
    metadata: dict,
    parquet_root: Path,
    metadata_order: list[str] | None = None,
) -> Path:
    """
    Compute the full path for a Parquet file.

    Structure: parquet_root / table_name / metadata_hierarchy / record_id.parquet

    Args:
        table_name: The variable type's table name (e.g., "sensor_reading")
        record_id: The version hash of the variable
        metadata: Dict of metadata key-value pairs
        parquet_root: Root folder for Parquet files
        metadata_order: Optional list of keys specifying folder order

    Returns:
        Full path to the Parquet file

    Example:
        >>> compute_parquet_path(
        ...     "sensor_reading",
        ...     "abc123def456",
        ...     {"subject": 1, "visit": 2},
        ...     Path("/data/parquet")
        ... )
        PosixPath('/data/parquet/sensor_reading/subject/1/visit/2/abc123def456.parquet')
    """
    folder_path = compute_folder_path(metadata, metadata_order)
    filename = f"{record_id}.parquet"
    return parquet_root / table_name / folder_path / filename


def _sanitize_path_component(value: str) -> str:
    """
    Sanitize a value for use as a path component.

    Replaces problematic characters with underscores.

    Args:
        value: The value to sanitize

    Returns:
        Sanitized string safe for use in file paths
    """
    # Replace path separators and other problematic characters
    unsafe_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\0']
    result = value
    for char in unsafe_chars:
        result = result.replace(char, '_')

    # Also handle leading/trailing dots and spaces
    result = result.strip('. ')

    # Empty result becomes underscore
    if not result:
        result = '_'

    return result


def parse_metadata_order(order_string: str) -> list[str]:
    """
    Parse a metadata order string into a list of keys.

    Args:
        order_string: Path-like string, e.g., "/visit/subject" or "visit/subject"

    Returns:
        List of keys in order, e.g., ["visit", "subject"]

    Example:
        >>> parse_metadata_order("/visit/subject")
        ['visit', 'subject']

        >>> parse_metadata_order("visit/subject")
        ['visit', 'subject']
    """
    # Remove leading/trailing slashes and split
    parts = order_string.strip('/').split('/')
    # Filter out empty parts
    return [p for p in parts if p]


def write_parquet(
    df: pd.DataFrame,
    path: Path,
    compression: str = "snappy",
) -> None:
    """
    Write a DataFrame to a Parquet file.

    Creates parent directories if they don't exist.

    Args:
        df: DataFrame to write
        path: Full path to the output file
        compression: Compression codec (default: "snappy")
    """
    # Ensure parent directories exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write Parquet file
    df.to_parquet(path, engine="pyarrow", compression=compression)


def read_parquet(path: Path) -> pd.DataFrame:
    """
    Read a DataFrame from a Parquet file.

    Args:
        path: Path to the Parquet file

    Returns:
        The DataFrame

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    return pd.read_parquet(path, engine="pyarrow")


def delete_parquet(path: Path, cleanup_empty_dirs: bool = True) -> bool:
    """
    Delete a Parquet file and optionally clean up empty parent directories.

    Args:
        path: Path to the Parquet file
        cleanup_empty_dirs: If True, remove empty parent directories up to parquet root

    Returns:
        True if file was deleted, False if it didn't exist
    """
    if not path.exists():
        return False

    path.unlink()

    if cleanup_empty_dirs:
        # Walk up and remove empty directories
        parent = path.parent
        while parent.name != "parquet" and parent != parent.parent:
            try:
                parent.rmdir()  # Only succeeds if empty
                parent = parent.parent
            except OSError:
                break  # Directory not empty

    return True


def list_parquet_files(
    parquet_root: Path,
    table_name: str | None = None,
    metadata_filter: dict | None = None,
) -> list[Path]:
    """
    List Parquet files in the storage hierarchy.

    Args:
        parquet_root: Root folder for Parquet files
        table_name: Optional filter by variable type (first-level folder)
        metadata_filter: Optional filter by metadata values (checks path components)

    Returns:
        List of paths to matching Parquet files
    """
    if not parquet_root.exists():
        return []

    # If table_name specified, search only in that folder
    if table_name is not None:
        search_root = parquet_root / table_name
        if not search_root.exists():
            return []
    else:
        search_root = parquet_root

    pattern = "**/*.parquet"
    all_files = list(search_root.glob(pattern))

    results = []
    for file_path in all_files:
        # Filter by metadata if specified
        if metadata_filter is not None:
            # Get the path relative to table folder
            try:
                if table_name is not None:
                    rel_path = file_path.relative_to(parquet_root / table_name)
                else:
                    # Extract table name from path (first component after parquet_root)
                    rel_to_root = file_path.relative_to(parquet_root)
                    # Skip the table_name folder for metadata matching
                    rel_path = Path(*rel_to_root.parts[1:]) if len(rel_to_root.parts) > 1 else Path(".")
            except ValueError:
                continue

            path_str = str(rel_path)
            matches = True
            for key, value in metadata_filter.items():
                expected = f"{key}/{_sanitize_path_component(str(value))}"
                if expected not in path_str:
                    matches = False
                    break
            if not matches:
                continue

        results.append(file_path)

    return sorted(results)


def extract_metadata_from_path(
    file_path: Path,
    parquet_root: Path,
) -> dict:
    """
    Extract metadata key-value pairs from a Parquet file path.

    Path structure: parquet_root / table_name / key / value / ... / record_id.parquet

    Args:
        file_path: Path to the Parquet file
        parquet_root: Root folder for Parquet files

    Returns:
        Dict of metadata extracted from path (excludes table_name)

    Example:
        >>> extract_metadata_from_path(
        ...     Path("/data/parquet/sensor_reading/subject/1/visit/2/abc123.parquet"),
        ...     Path("/data/parquet")
        ... )
        {'subject': '1', 'visit': '2'}
    """
    try:
        rel_path = file_path.relative_to(parquet_root)
    except ValueError:
        return {}

    # Path parts: table_name, key, value, key, value, ..., filename
    parts = list(rel_path.parts)

    if len(parts) < 2:
        return {}  # Just table_name and filename, no metadata

    # Skip first part (table_name) and last part (filename)
    metadata_parts = parts[1:-1]

    metadata = {}
    for i in range(0, len(metadata_parts) - 1, 2):
        key = metadata_parts[i]
        value = metadata_parts[i + 1]
        metadata[key] = value

    return metadata


def extract_table_name_from_path(
    file_path: Path,
    parquet_root: Path,
) -> str | None:
    """
    Extract the table name from a Parquet file path.

    Path structure: parquet_root / table_name / ... / record_id.parquet

    Args:
        file_path: Path to the Parquet file
        parquet_root: Root folder for Parquet files

    Returns:
        The table name, or None if path doesn't match expected structure

    Example:
        >>> extract_table_name_from_path(
        ...     Path("/data/parquet/sensor_reading/subject/1/abc123.parquet"),
        ...     Path("/data/parquet")
        ... )
        'sensor_reading'
    """
    try:
        rel_path = file_path.relative_to(parquet_root)
    except ValueError:
        return None

    parts = rel_path.parts
    if len(parts) < 1:
        return None

    return parts[0]


def extract_record_id_from_filename(filename: str) -> str | None:
    """
    Extract the record_id from a Parquet filename.

    Filename format: <record_id>.parquet

    Args:
        filename: Filename like "abc123def456.parquet"

    Returns:
        The record_id, or None if filename doesn't match expected format

    Example:
        >>> extract_record_id_from_filename("abc123def456.parquet")
        'abc123def456'
    """
    if not filename.endswith(".parquet"):
        return None

    # Remove .parquet extension
    return filename[:-8] if len(filename) > 8 else None
