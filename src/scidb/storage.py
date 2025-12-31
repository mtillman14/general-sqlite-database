"""SQLite BLOB serialization helpers."""

import pickle
import sqlite3

import pandas as pd


def serialize_dataframe(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to bytes for BLOB storage."""
    return pickle.dumps(df, protocol=4)


def deserialize_dataframe(blob: bytes) -> pd.DataFrame:
    """Deserialize bytes back to a DataFrame."""
    return pickle.loads(blob)


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
