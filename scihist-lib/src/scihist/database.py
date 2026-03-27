"""SciHist database configuration — wraps scidb.configure_database with cache backend registration."""

from typing import Any


def configure_database(
    db_path: str,
    schema_keys: list[str] | None = None,
    **kwargs,
) -> Any:
    """Configure the active database and register it as the lineage cache backend.

    This is the scihist wrapper around scidb.configure_database(). In addition
    to opening the DuckDB-backed database, it registers the database as the
    cache backend so that lineage-tracked computations can look up previously
    computed results.

    Args:
        db_path: Path to the DuckDB database file.
        schema_keys: List of metadata keys that form the dataset schema.
        **kwargs: Additional keyword arguments forwarded to scidb.configure_database().

    Returns:
        The configured DatabaseManager instance.
    """
    from scidb import configure_database as _scidb_configure
    from scilineage import configure_backend

    db = _scidb_configure(db_path, schema_keys, **kwargs)
    configure_backend(db)
    return db


def find_by_lineage(invocation) -> list | None:
    """Find output values by computation lineage.

    Given a LineageFcnInvocation (function + inputs), finds any previously
    computed outputs that match by querying the _lineage table via the active
    database.

    Args:
        invocation: The LineageFcnInvocation computation to look up.

    Returns:
        List of output values if found, None otherwise.
    """
    from scidb.database import get_database

    db = get_database()
    lineage_hash = invocation.compute_lineage_hash()
    return db.find_by_lineage_hash(lineage_hash)
