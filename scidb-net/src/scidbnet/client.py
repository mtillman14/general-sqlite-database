"""RemoteDatabaseManager — drop-in replacement for DatabaseManager over HTTP.

All DatabaseManager public methods are mirrored. ThunkOutput handling and
to_db()/from_db() run client-side; the server stores/returns raw data.
"""

import json
import struct
from typing import Any, Type

import httpx
import numpy as np
import pandas as pd

from scidb.variable import BaseVariable

from .exceptions import NetworkError, SerializationError, ServerError
from .serialization import (
    decode_envelope,
    decode_response,
    deserialize_data,
    encode_save_request,
    serialize_data,
)


class RemoteDatabaseManager:
    """Client that mirrors DatabaseManager's public API over HTTP.

    Usage:
        from scidbnet import configure_remote_database

        configure_remote_database("http://localhost:8000")
        # Now BaseVariable.save() / .load() / thunk caching all work remotely.
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self._registered_types: dict[str, Type[BaseVariable]] = {}

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"/api/v1/{path}"

    def _post_json(self, path: str, data: dict) -> dict:
        """POST JSON, return parsed JSON response."""
        try:
            resp = self._client.post(self._url(path), json=data)
        except httpx.HTTPError as exc:
            raise NetworkError(f"HTTP request failed: {exc}") from exc
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = {"error": resp.text}
            raise ServerError(
                detail.get("error", resp.text),
                status_code=resp.status_code,
            )
        return resp.json()

    def _post_binary(self, path: str, body: bytes) -> httpx.Response:
        """POST binary data, return raw response."""
        try:
            resp = self._client.post(
                self._url(path),
                content=body,
                headers={"Content-Type": "application/octet-stream"},
            )
        except httpx.HTTPError as exc:
            raise NetworkError(f"HTTP request failed: {exc}") from exc
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = {"error": resp.text}
            raise ServerError(
                detail.get("error", resp.text),
                status_code=resp.status_code,
            )
        return resp

    @staticmethod
    def _has_custom_serialization(variable_class: type) -> bool:
        """Check if a BaseVariable subclass overrides to_db or from_db."""
        return "to_db" in variable_class.__dict__ or "from_db" in variable_class.__dict__

    def _response_to_variable(
        self,
        variable_class: Type[BaseVariable],
        header: dict,
        body: bytes,
    ) -> BaseVariable:
        """Reconstruct a BaseVariable from an envelope header + body."""
        raw_data = deserialize_data(
            {k: v for k, v in header.items() if not k.startswith("_")},
            body,
        )

        # If the type has custom serialization, raw_data is a DataFrame
        # that needs from_db()
        if self._has_custom_serialization(variable_class) and isinstance(
            raw_data, pd.DataFrame
        ):
            # Strip internal columns before passing to from_db
            internal_cols = [
                c for c in raw_data.columns if c.startswith("_")
            ]
            if internal_cols:
                raw_data = raw_data.drop(columns=internal_cols)
            raw_data = variable_class.from_db(raw_data)

        instance = variable_class(raw_data)
        instance._record_id = header.get("_record_id")
        instance._metadata = header.get("_metadata")
        instance._content_hash = header.get("_content_hash")
        instance._lineage_hash = header.get("_lineage_hash")
        return instance

    # -----------------------------------------------------------------
    # Public API (mirrors DatabaseManager)
    # -----------------------------------------------------------------

    def register(self, variable_class: Type[BaseVariable]) -> None:
        """Register a variable type with the remote server."""
        type_name = variable_class.__name__
        self._registered_types[type_name] = variable_class
        self._post_json("register", {
            "type_name": type_name,
            "table_name": variable_class.table_name(),
            "schema_version": variable_class.schema_version,
            "has_custom_serialization": self._has_custom_serialization(variable_class),
        })

    def save_variable(
        self,
        variable_class: Type[BaseVariable],
        data: Any,
        index: Any = None,
        **metadata,
    ) -> str:
        """Save data as a variable, handling ThunkOutput extraction client-side."""
        from scidb.thunk import ThunkOutput
        from scidb.lineage import extract_lineage, find_unsaved_variables, get_raw_value

        lineage = None
        lineage_hash = None
        raw_data = None

        if isinstance(data, ThunkOutput):
            lineage = extract_lineage(data)
            lineage_hash = data.pipeline_thunk.compute_lineage_hash()
            raw_data = get_raw_value(data)
        elif isinstance(data, BaseVariable):
            raw_data = data.data
            lineage_hash = data._lineage_hash
        else:
            raw_data = data

        # Client-side to_db() for custom serialization
        has_custom = self._has_custom_serialization(variable_class)
        instance = variable_class(raw_data)
        if has_custom:
            send_data = instance.to_db()
            if index is not None:
                index_list = list(index) if not isinstance(index, list) else index
                send_data.index = index_list
        else:
            send_data = raw_data

        meta = {
            "type_name": variable_class.__name__,
            "metadata": metadata,
            "lineage": lineage.to_dict() if lineage else None,
            "lineage_hash": lineage_hash,
            "index": list(index) if index is not None else None,
            "has_custom_serialization": has_custom,
        }

        body = encode_save_request(meta, send_data)
        resp = self._post_binary("save", body)
        result = resp.json()
        return result["record_id"]

    def save(
        self,
        variable: BaseVariable,
        metadata: dict,
        lineage: Any = None,
        lineage_hash: str | None = None,
        index: Any = None,
    ) -> str:
        """Save a variable instance to the remote database."""
        has_custom = self._has_custom_serialization(type(variable))
        if has_custom:
            send_data = variable.to_db()
            if index is not None:
                index_list = list(index) if not isinstance(index, list) else index
                send_data.index = index_list
        else:
            send_data = variable.data

        meta = {
            "type_name": variable.__class__.__name__,
            "metadata": metadata,
            "lineage": lineage.to_dict() if lineage and hasattr(lineage, "to_dict") else lineage,
            "lineage_hash": lineage_hash,
            "index": list(index) if index is not None else None,
            "has_custom_serialization": has_custom,
        }

        body = encode_save_request(meta, send_data)
        resp = self._post_binary("save", body)
        result = resp.json()
        return result["record_id"]

    def load(
        self,
        variable_class: Type[BaseVariable],
        metadata: dict,
        version: str = "latest",
        loc: Any = None,
        iloc: Any = None,
    ) -> BaseVariable:
        """Load a single variable from the remote database."""
        req = {
            "type_name": variable_class.__name__,
            "metadata": metadata,
            "version": version,
        }
        if loc is not None:
            req["loc"] = loc if isinstance(loc, list) else [loc]
        if iloc is not None:
            req["iloc"] = iloc if isinstance(iloc, list) else [iloc]

        resp = self._post_binary("load", json.dumps(req).encode("utf-8"))
        header, body = decode_envelope(resp.content)
        return self._response_to_variable(variable_class, header, body)

    def load_all(
        self,
        variable_class: Type[BaseVariable],
        metadata: dict,
    ):
        """Load all matching variables as a generator."""
        req = {
            "type_name": variable_class.__name__,
            "metadata": metadata,
        }
        resp = self._post_binary("load_all", json.dumps(req).encode("utf-8"))
        data = resp.content

        if len(data) < 4:
            return

        count = struct.unpack(">I", data[:4])[0]
        offset = 4
        for _ in range(count):
            if offset + 4 > len(data):
                break
            part_len = struct.unpack(">I", data[offset : offset + 4])[0]
            offset += 4
            envelope = data[offset : offset + part_len]
            offset += part_len

            header, body = decode_envelope(envelope)
            yield self._response_to_variable(variable_class, header, body)

    def list_versions(
        self,
        variable_class: Type[BaseVariable],
        **metadata,
    ) -> list[dict]:
        """List all versions at a schema location."""
        result = self._post_json("list_versions", {
            "type_name": variable_class.__name__,
            "metadata": metadata,
        })
        return result["versions"]

    def get_provenance(
        self,
        variable_class: Type[BaseVariable],
        version: str | None = None,
        **metadata,
    ) -> dict | None:
        """Get provenance of a variable."""
        result = self._post_json("provenance", {
            "type_name": variable_class.__name__,
            "version": version,
            "metadata": metadata,
        })
        return result["provenance"]

    def get_provenance_by_schema(self, **schema_keys) -> list[dict]:
        """Get all provenance records matching schema keys."""
        result = self._post_json("provenance_by_schema", {
            "schema_keys": schema_keys,
        })
        return result["records"]

    def get_pipeline_structure(self) -> list[dict]:
        """Get the abstract pipeline structure."""
        try:
            resp = self._client.get(self._url("pipeline_structure"))
        except httpx.HTTPError as exc:
            raise NetworkError(f"HTTP request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise ServerError(resp.text, status_code=resp.status_code)
        return resp.json()["structure"]

    def has_lineage(self, record_id: str) -> bool:
        """Check if a variable has lineage information."""
        result = self._post_json("has_lineage", {"record_id": record_id})
        return result["has_lineage"]

    def save_ephemeral_lineage(
        self,
        ephemeral_id: str,
        variable_type: str,
        lineage: Any,
        user_id: str | None = None,
        schema_keys: dict | None = None,
    ) -> None:
        """Save an ephemeral lineage record."""
        lineage_dict = lineage.to_dict() if hasattr(lineage, "to_dict") else lineage
        self._post_json("save_ephemeral_lineage", {
            "ephemeral_id": ephemeral_id,
            "variable_type": variable_type,
            "lineage": lineage_dict,
            "user_id": user_id,
            "schema_keys": schema_keys,
        })

    def export_to_csv(
        self,
        variable_class: Type[BaseVariable],
        path: str,
        **metadata,
    ) -> int:
        """Export matching variables to CSV (server-side)."""
        result = self._post_json("export_to_csv", {
            "type_name": variable_class.__name__,
            "path": path,
            "metadata": metadata,
        })
        return result["count"]

    def find_by_lineage(self, pipeline_thunk) -> list | None:
        """Find cached outputs by computation lineage.

        This is called by Thunk.__call__ for cache lookup.
        The client computes the lineage hash locally and sends just the hash.
        """
        lineage_hash = pipeline_thunk.compute_lineage_hash()

        resp = self._post_binary(
            "find_by_lineage",
            json.dumps({"lineage_hash": lineage_hash}).encode("utf-8"),
        )

        if resp.status_code == 204 or not resp.content:
            return None

        data = resp.content
        if len(data) < 4:
            return None

        count = struct.unpack(">I", data[:4])[0]
        offset = 4
        results = []
        for _ in range(count):
            if offset + 4 > len(data):
                return None
            part_len = struct.unpack(">I", data[offset : offset + 4])[0]
            offset += 4
            envelope = data[offset : offset + part_len]
            offset += part_len

            header, body = decode_envelope(envelope)
            # Extract type name to apply from_db if needed
            value = deserialize_data(
                {k: v for k, v in header.items() if not k.startswith("_")},
                body,
            )
            results.append(value)

        return results if results else None

    def close(self):
        """Close the remote connection."""
        try:
            self._post_json("close", {})
        except Exception:
            pass
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
