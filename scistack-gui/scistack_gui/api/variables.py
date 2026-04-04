"""
GET /api/variables/{variable_name}/records

Returns all records for a named variable type, with schema key values and
branch_params per record, plus a variant summary grouped by branch_params.
"""

import json
from fastapi import APIRouter, Depends, HTTPException
from scidb.database import DatabaseManager
from scistack_gui.db import get_db

router = APIRouter()


def _format_variant_label(branch_params: dict) -> str:
    """
    Produce a concise human-readable label from branch_params.

    branch_params keys are namespaced as "fn_name.param" (for constants) or bare
    names (for dynamic discriminators). Strip the function prefix when all keys
    share the same function, so the display stays compact.
    """
    if not branch_params:
        return "(raw)"

    # Collect key=value pairs, stripping common fn prefix for readability.
    parts = []
    for k, v in sorted(branch_params.items()):
        short_k = k.split(".")[-1] if "." in k else k
        parts.append(f"{short_k}={v}")
    return ", ".join(parts)


@router.get("/variables/{variable_name}/records")
def get_variable_records(variable_name: str, db: DatabaseManager = Depends(get_db)):
    """
    Return all records for a variable type with schema key values and variant info.

    Response shape:
      {
        "schema_keys": ["subject", "session"],
        "records": [
          {"subject": "1", "session": "pre", "branch_params": {...}, "variant_label": "..."},
          ...
        ],
        "variants": [
          {"label": "...", "branch_params": {...}, "record_count": 4},
          ...
        ]
      }
    """
    schema_keys: list[str] = db._duck.dataset_schema

    # Dynamically build the SELECT clause for schema key columns.
    schema_select = ", ".join(f's."{k}"' for k in schema_keys)
    if schema_select:
        schema_select += ", "

    query = f"""
        WITH latest AS (
            SELECT record_id, schema_id, branch_params
            FROM _record_metadata
            WHERE variable_name = $1
              AND excluded = FALSE
            QUALIFY ROW_NUMBER() OVER (PARTITION BY record_id ORDER BY timestamp DESC) = 1
        )
        SELECT {schema_select}l.branch_params
        FROM latest l
        LEFT JOIN _schema s ON l.schema_id = s.schema_id
        ORDER BY {", ".join(f's."{k}"' for k in schema_keys) or "l.branch_params"}
    """

    try:
        rows = db._duck._fetchall(query, [variable_name])
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    col_names = schema_keys + ["branch_params"]

    records = []
    for row in rows:
        row_dict = dict(zip(col_names, row))
        raw_bp = row_dict.get("branch_params") or "{}"
        bp = json.loads(raw_bp) if isinstance(raw_bp, str) else (raw_bp or {})
        records.append({
            **{k: str(row_dict[k]) if row_dict[k] is not None else None for k in schema_keys},
            "branch_params": bp,
            "variant_label": _format_variant_label(bp),
        })

    # Build variant summary: group by branch_params JSON (canonical sort).
    variant_map: dict[str, dict] = {}
    for rec in records:
        key = json.dumps(rec["branch_params"], sort_keys=True)
        if key not in variant_map:
            variant_map[key] = {
                "label": rec["variant_label"],
                "branch_params": rec["branch_params"],
                "record_count": 0,
            }
        variant_map[key]["record_count"] += 1

    variants = list(variant_map.values())

    return {
        "schema_keys": schema_keys,
        "records": records,
        "variants": variants,
    }
