"""SQLite BLOB serialization helpers."""

import io
import sqlite3

import pandas as pd


def serialize_dataframe(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to bytes for BLOB storage.

    Uses Parquet format for long-term archival stability.
    Parquet is cross-language, schema-preserving, and not tied to Python versions.
    """
    buffer = io.BytesIO()
    df.to_parquet(buffer, engine="pyarrow")
    return buffer.getvalue()


def deserialize_dataframe(blob: bytes) -> pd.DataFrame:
    """Deserialize bytes back to a DataFrame."""
    buffer = io.BytesIO(blob)
    return pd.read_parquet(buffer, engine="pyarrow")


def adapt_dataframe(df: pd.DataFrame) -> bytes:
    """SQLite adapter for DataFrame."""
    return serialize_dataframe(df)


def convert_dataframe(blob: bytes) -> pd.DataFrame:
    """SQLite converter for DataFrame."""
    return deserialize_dataframe(blob)


def register_adapters():
    """Register custom SQLite adapters/converters."""
    sqlite3.register_adapter(pd.DataFrame, adapt_dataframe)
    sqlite3.register_converter("DATAFRAME", convert_dataframe)
