"""Serialization layer: Arrow IPC for DataFrames/arrays, JSON for scalars/dicts.

Wire format (envelope):
    4-byte header length (big-endian) | JSON header | body bytes

Header describes the format:
    {"format": "dataframe"}
    {"format": "numpy", "dtype": "float64", "shape": [3, 4]}
    {"format": "json_scalar", "python_type": "int"}
    {"format": "json_value"}
    {"format": "none"}

Multi-value responses (load_all, find_by_lineage):
    4-byte count | [4-byte part_len | envelope]...

Save requests:
    4-byte meta_len | JSON metadata | data envelope
"""

import json
import struct
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from .exceptions import SerializationError


# ---------------------------------------------------------------------------
# Core serialization: Python object <-> (header_dict, body_bytes)
# ---------------------------------------------------------------------------

def serialize_data(data: Any) -> tuple[dict, bytes]:
    """Serialize a Python object to (header_dict, body_bytes).

    Supported types:
        - None -> ("none", b"")
        - pd.DataFrame -> ("dataframe", Arrow IPC bytes)
        - np.ndarray -> ("numpy", Arrow IPC bytes via single-column table)
        - scalar / dict / list -> ("json_scalar" or "json_value", JSON bytes)
    """
    if data is None:
        return {"format": "none"}, b""

    if isinstance(data, pd.DataFrame):
        table = pa.Table.from_pandas(data, preserve_index=True)
        sink = pa.BufferOutputStream()
        writer = ipc.new_stream(sink, table.schema)
        writer.write_table(table)
        writer.close()
        return {"format": "dataframe"}, sink.getvalue().to_pybytes()

    if isinstance(data, np.ndarray):
        header = {
            "format": "numpy",
            "dtype": str(data.dtype),
            "shape": list(data.shape),
        }
        flat = data.flatten()
        table = pa.table({"v": flat})
        sink = pa.BufferOutputStream()
        writer = ipc.new_stream(sink, table.schema)
        writer.write_table(table)
        writer.close()
        return header, sink.getvalue().to_pybytes()

    # Scalars and JSON-compatible types
    if isinstance(data, (int, float, bool, str)):
        header = {"format": "json_scalar", "python_type": type(data).__name__}
        body = json.dumps(data).encode("utf-8")
        return header, body

    # dicts, lists, etc.
    header = {"format": "json_value"}
    body = json.dumps(data).encode("utf-8")
    return header, body


def deserialize_data(header: dict, body: bytes) -> Any:
    """Deserialize (header_dict, body_bytes) back to a Python object."""
    fmt = header.get("format")

    if fmt == "none":
        return None

    if fmt == "dataframe":
        reader = ipc.open_stream(body)
        table = reader.read_all()
        return table.to_pandas()

    if fmt == "numpy":
        reader = ipc.open_stream(body)
        table = reader.read_all()
        flat = table.column("v").to_numpy()
        dtype = np.dtype(header["dtype"])
        shape = tuple(header["shape"])
        return flat.astype(dtype).reshape(shape)

    if fmt == "json_scalar":
        value = json.loads(body.decode("utf-8"))
        python_type = header.get("python_type")
        if python_type == "int":
            return int(value)
        if python_type == "float":
            return float(value)
        if python_type == "bool":
            return bool(value)
        if python_type == "str":
            return str(value)
        return value

    if fmt == "json_value":
        return json.loads(body.decode("utf-8"))

    raise SerializationError(f"Unknown format: {fmt}")


# ---------------------------------------------------------------------------
# Envelope encoding: header + body -> single bytes blob
# ---------------------------------------------------------------------------

def encode_envelope(header: dict, body: bytes) -> bytes:
    """Encode (header, body) into a single envelope bytes blob.

    Format: 4-byte header_len (big-endian) | JSON header bytes | body bytes
    """
    header_bytes = json.dumps(header).encode("utf-8")
    return struct.pack(">I", len(header_bytes)) + header_bytes + body


def decode_envelope(data: bytes) -> tuple[dict, bytes]:
    """Decode an envelope bytes blob into (header, body)."""
    if len(data) < 4:
        raise SerializationError("Envelope too short to contain header length")
    header_len = struct.unpack(">I", data[:4])[0]
    if len(data) < 4 + header_len:
        raise SerializationError("Envelope truncated: missing header bytes")
    header = json.loads(data[4 : 4 + header_len].decode("utf-8"))
    body = data[4 + header_len :]
    return header, body


# ---------------------------------------------------------------------------
# Convenience: serialize/deserialize full envelope in one call
# ---------------------------------------------------------------------------

def encode_response(data: Any) -> bytes:
    """Serialize a Python object into a complete envelope."""
    header, body = serialize_data(data)
    return encode_envelope(header, body)


def decode_response(data: bytes) -> Any:
    """Deserialize a complete envelope back to a Python object."""
    header, body = decode_envelope(data)
    return deserialize_data(header, body)


# ---------------------------------------------------------------------------
# Multi-value encoding (for load_all, find_by_lineage)
# ---------------------------------------------------------------------------

def encode_multi(items: list[Any]) -> bytes:
    """Encode a list of objects into a packed multi-value blob.

    Format: 4-byte count | [4-byte part_len | envelope]...
    """
    parts: list[bytes] = []
    for item in items:
        envelope = encode_response(item)
        parts.append(struct.pack(">I", len(envelope)) + envelope)
    return struct.pack(">I", len(items)) + b"".join(parts)


def decode_multi(data: bytes) -> list[Any]:
    """Decode a packed multi-value blob back to a list of objects."""
    if len(data) < 4:
        raise SerializationError("Multi-value blob too short")
    count = struct.unpack(">I", data[:4])[0]
    offset = 4
    results: list[Any] = []
    for _ in range(count):
        if offset + 4 > len(data):
            raise SerializationError("Multi-value blob truncated")
        part_len = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        envelope = data[offset : offset + part_len]
        results.append(decode_response(envelope))
        offset += part_len
    return results


# ---------------------------------------------------------------------------
# Save request encoding (metadata + data envelope)
# ---------------------------------------------------------------------------

def encode_save_request(meta: dict, data: Any) -> bytes:
    """Encode a save request: JSON metadata + data envelope.

    Format: 4-byte meta_len | JSON metadata bytes | data envelope bytes
    """
    meta_bytes = json.dumps(meta).encode("utf-8")
    data_envelope = encode_response(data)
    return struct.pack(">I", len(meta_bytes)) + meta_bytes + data_envelope


def decode_save_request(data: bytes) -> tuple[dict, Any]:
    """Decode a save request into (metadata_dict, deserialized_data)."""
    if len(data) < 4:
        raise SerializationError("Save request too short")
    meta_len = struct.unpack(">I", data[:4])[0]
    if len(data) < 4 + meta_len:
        raise SerializationError("Save request truncated: missing metadata")
    meta = json.loads(data[4 : 4 + meta_len].decode("utf-8"))
    data_envelope = data[4 + meta_len :]
    value = decode_response(data_envelope)
    return meta, value
