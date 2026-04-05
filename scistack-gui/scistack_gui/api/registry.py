"""
GET /api/registry

Returns all pipeline functions and variable types known to the server,
sourced from the user module loaded at startup.
"""

import logging

from fastapi import APIRouter
from scistack_gui import registry
from scistack_gui.api import ws
from scidb import BaseVariable

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/registry")
def get_registry() -> dict:
    return {
        "functions": sorted(registry._functions.keys()),
        "variables": sorted(BaseVariable._all_subclasses.keys()),
    }


@router.post("/refresh")
async def refresh_registry() -> dict:
    """Re-import the user module from disk and refresh the function/variable registry."""
    try:
        result = registry.refresh_module()
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        logger.exception("Failed to refresh module")
        return {"ok": False, "error": f"Import error: {e}"}
    await ws.broadcast({"type": "dag_updated"})
    return {"ok": True, **result}
