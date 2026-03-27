"""Backend registry for scilineage cache lookups.

Provides a module-level slot for a single cache backend. Higher-level packages
(scihist, scidb-net) register their database or network client here once at
startup. End users do not interact with this directly.
"""

_backend = None


def configure_backend(backend) -> None:
    """Register a cache backend.

    Called by scihist.configure_database() or scidb-net at startup.
    Not intended to be called by end users directly.
    """
    global _backend
    _backend = backend


def _clear_backend() -> None:
    """Reset the backend to None. For use in tests."""
    global _backend
    _backend = None


def _get_backend():
    """Return the current backend, or None if not configured."""
    return _backend
