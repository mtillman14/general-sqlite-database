"""SciLineage: function-source hashing utilities.

What remains of scilineage after the lineage-wrapper system (``@lineage_fcn`` /
``LineageFcnResult`` / input classification / rerun cache) was removed in favor
of scidb's ``@scistack`` + bipartite provenance graph: the bytecode/AST-based
function hashing that scidb uses for function identity in the graph.

    from scilineage import compute_function_hash, canonical_hash
"""

from .hashing import (
    canonical_hash,
    compute_function_hash,
    compute_function_hash_with_sources,
)

__version__ = "0.1.0"

__all__ = [
    "canonical_hash",
    "compute_function_hash",
    "compute_function_hash_with_sources",
]
