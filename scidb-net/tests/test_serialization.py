"""Round-trip unit tests for the serialization layer."""

import json

import numpy as np
import pandas as pd
import pytest

from scidbnet.serialization import (
    decode_envelope,
    decode_multi,
    decode_response,
    decode_save_request,
    deserialize_data,
    encode_envelope,
    encode_multi,
    encode_response,
    encode_save_request,
    serialize_data,
)
from scidbnet.exceptions import SerializationError


# ---------------------------------------------------------------------------
# serialize_data / deserialize_data round-trips
# ---------------------------------------------------------------------------

class TestSerializeRoundTrip:
    def test_none(self):
        header, body = serialize_data(None)
        assert header["format"] == "none"
        assert body == b""
        assert deserialize_data(header, body) is None

    def test_int(self):
        header, body = serialize_data(42)
        assert header["format"] == "json_scalar"
        assert header["python_type"] == "int"
        result = deserialize_data(header, body)
        assert result == 42
        assert isinstance(result, int)

    def test_float(self):
        header, body = serialize_data(3.14)
        result = deserialize_data(header, body)
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_bool(self):
        for val in [True, False]:
            header, body = serialize_data(val)
            result = deserialize_data(header, body)
            assert result == val
            assert isinstance(result, bool)

    def test_string(self):
        header, body = serialize_data("hello world")
        result = deserialize_data(header, body)
        assert result == "hello world"
        assert isinstance(result, str)

    def test_dict(self):
        original = {"a": 1, "b": [2, 3], "c": {"nested": True}}
        header, body = serialize_data(original)
        assert header["format"] == "json_value"
        result = deserialize_data(header, body)
        assert result == original

    def test_list(self):
        original = [1, "two", 3.0, None]
        header, body = serialize_data(original)
        result = deserialize_data(header, body)
        assert result == original

    def test_numpy_1d(self):
        original = np.array([1.0, 2.0, 3.0, 4.0])
        header, body = serialize_data(original)
        assert header["format"] == "numpy"
        assert header["shape"] == [4]
        result = deserialize_data(header, body)
        np.testing.assert_array_equal(result, original)
        assert result.dtype == original.dtype

    def test_numpy_2d(self):
        original = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        header, body = serialize_data(original)
        assert header["shape"] == [2, 3]
        result = deserialize_data(header, body)
        np.testing.assert_array_equal(result, original)
        assert result.shape == (2, 3)

    def test_numpy_int64(self):
        original = np.arange(10, dtype=np.int64)
        header, body = serialize_data(original)
        result = deserialize_data(header, body)
        np.testing.assert_array_equal(result, original)

    def test_dataframe(self):
        original = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        header, body = serialize_data(original)
        assert header["format"] == "dataframe"
        result = deserialize_data(header, body)
        pd.testing.assert_frame_equal(result, original)

    def test_dataframe_with_string_columns(self):
        original = pd.DataFrame({"name": ["a", "b"], "value": [1, 2]})
        header, body = serialize_data(original)
        result = deserialize_data(header, body)
        pd.testing.assert_frame_equal(result, original)

    def test_unknown_format_raises(self):
        with pytest.raises(SerializationError, match="Unknown format"):
            deserialize_data({"format": "unknown_xyz"}, b"")


# ---------------------------------------------------------------------------
# Envelope encoding
# ---------------------------------------------------------------------------

class TestEnvelope:
    def test_round_trip(self):
        header = {"format": "json_scalar", "python_type": "int"}
        body = b"42"
        encoded = encode_envelope(header, body)
        dec_header, dec_body = decode_envelope(encoded)
        assert dec_header == header
        assert dec_body == body

    def test_too_short_raises(self):
        with pytest.raises(SerializationError):
            decode_envelope(b"\x00")

    def test_truncated_header_raises(self):
        with pytest.raises(SerializationError):
            decode_envelope(b"\x00\x00\x00\xff")


# ---------------------------------------------------------------------------
# Full encode_response / decode_response
# ---------------------------------------------------------------------------

class TestResponseCodec:
    def test_int_round_trip(self):
        assert decode_response(encode_response(42)) == 42

    def test_array_round_trip(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = decode_response(encode_response(arr))
        np.testing.assert_array_equal(result, arr)

    def test_dataframe_round_trip(self):
        df = pd.DataFrame({"a": [10, 20]})
        result = decode_response(encode_response(df))
        pd.testing.assert_frame_equal(result, df)

    def test_none_round_trip(self):
        assert decode_response(encode_response(None)) is None


# ---------------------------------------------------------------------------
# Multi-value encoding
# ---------------------------------------------------------------------------

class TestMultiCodec:
    def test_round_trip(self):
        items = [42, np.array([1.0, 2.0]), "hello"]
        encoded = encode_multi(items)
        decoded = decode_multi(encoded)
        assert decoded[0] == 42
        np.testing.assert_array_equal(decoded[1], items[1])
        assert decoded[2] == "hello"

    def test_empty(self):
        encoded = encode_multi([])
        decoded = decode_multi(encoded)
        assert decoded == []

    def test_too_short_raises(self):
        with pytest.raises(SerializationError):
            decode_multi(b"\x00")


# ---------------------------------------------------------------------------
# Save request encoding
# ---------------------------------------------------------------------------

class TestSaveRequestCodec:
    def test_round_trip_with_array(self):
        meta = {"type_name": "ArrayVar", "metadata": {"subject": 1}}
        arr = np.array([1.0, 2.0, 3.0])
        encoded = encode_save_request(meta, arr)
        dec_meta, dec_data = decode_save_request(encoded)
        assert dec_meta == meta
        np.testing.assert_array_equal(dec_data, arr)

    def test_round_trip_with_scalar(self):
        meta = {"type_name": "ScalarVar", "metadata": {}}
        encoded = encode_save_request(meta, 99)
        dec_meta, dec_data = decode_save_request(encoded)
        assert dec_meta == meta
        assert dec_data == 99

    def test_too_short_raises(self):
        with pytest.raises(SerializationError):
            decode_save_request(b"\x00")
