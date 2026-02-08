"""Pydantic request/response models shared by client and server."""

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    type_name: str
    table_name: str
    schema_version: int
    has_custom_serialization: bool


class RegisterResponse(BaseModel):
    ok: bool


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

class SaveMeta(BaseModel):
    """JSON metadata sent alongside save binary payload."""
    type_name: str
    metadata: dict
    lineage: dict | None = None
    lineage_hash: str | None = None
    index: list | None = None
    has_custom_serialization: bool


class SaveResponse(BaseModel):
    record_id: str


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

class LoadRequest(BaseModel):
    type_name: str
    metadata: dict
    version: str = "latest"
    loc: list | None = None
    iloc: list | None = None


class LoadAllRequest(BaseModel):
    type_name: str
    metadata: dict


# ---------------------------------------------------------------------------
# List / Provenance
# ---------------------------------------------------------------------------

class ListVersionsRequest(BaseModel):
    type_name: str
    metadata: dict


class ListVersionsResponse(BaseModel):
    versions: list[dict]


class ProvenanceRequest(BaseModel):
    type_name: str
    version: str | None = None
    metadata: dict


class ProvenanceResponse(BaseModel):
    provenance: dict | None


class ProvenanceBySchemaRequest(BaseModel):
    schema_keys: dict


class ProvenanceBySchemaResponse(BaseModel):
    records: list[dict]


# ---------------------------------------------------------------------------
# Pipeline / Lineage
# ---------------------------------------------------------------------------

class PipelineStructureResponse(BaseModel):
    structure: list[dict]


class HasLineageRequest(BaseModel):
    record_id: str


class HasLineageResponse(BaseModel):
    has_lineage: bool


class SaveEphemeralLineageRequest(BaseModel):
    ephemeral_id: str
    variable_type: str
    lineage: dict
    user_id: str | None = None
    schema_keys: dict | None = None


class SaveEphemeralLineageResponse(BaseModel):
    ok: bool


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class ExportToCsvRequest(BaseModel):
    type_name: str
    path: str
    metadata: dict


class ExportToCsvResponse(BaseModel):
    count: int


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class FindByLineageRequest(BaseModel):
    lineage_hash: str


# ---------------------------------------------------------------------------
# Health / Close
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str


class CloseResponse(BaseModel):
    ok: bool


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
