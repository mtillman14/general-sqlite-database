"""Python bridge for MATLAB-SciDB integration.

Provides proxy classes that satisfy the duck-typing contracts of
thunk-lib's Thunk, PipelineThunk, and ThunkOutput classes.  This allows
MATLAB functions to participate fully in the lineage / caching system
without any changes to existing Python packages.

The key insight is that every Python function that touches these objects
uses duck-typing (attribute access), not isinstance checks on Thunk or
PipelineThunk.  ThunkOutput *is* instantiated directly from thunk-lib,
so isinstance checks in save_variable() pass naturally.

Duck-typing contracts satisfied
-------------------------------
MatlabThunk provides:
    .hash            str   (64-char hex, same algorithm as Thunk.__init__)
    .fcn.__name__    str   (used by extract_lineage)
    .unpack_output   bool
    .unwrap          bool
    .pipeline_thunks tuple

MatlabPipelineThunk provides:
    .thunk           MatlabThunk
    .inputs          dict[str, Any]
    .outputs         tuple
    .unwrap          bool
    .hash            property -> compute_lineage_hash()
    .compute_lineage_hash()  str  (reuses classify_inputs from thunk-lib)
"""

from hashlib import sha256

from thunk.inputs import classify_inputs

STRING_REPR_DELIMITER = "-"


# ---------------------------------------------------------------------------
# Proxy classes
# ---------------------------------------------------------------------------


class _FunctionProxy:
    """Minimal proxy so that ``pt.thunk.fcn.__name__`` works in extract_lineage."""

    def __init__(self, name: str):
        self.__name__ = name


class MatlabThunk:
    """Proxy for a MATLAB function in the thunk lineage system.

    Satisfies the same duck-typing contract as ``thunk.core.Thunk`` for
    every consumer that reads ``.hash``, ``.fcn.__name__``, etc.

    Parameters
    ----------
    source_hash : str
        SHA-256 hex digest of the MATLAB function source code.
    function_name : str
        Human-readable function name (used in lineage records).
    unpack_output : bool
        Whether the function returns multiple outputs.
    """

    def __init__(
        self,
        source_hash: str,
        function_name: str,
        unpack_output: bool = False,
    ):
        self.fcn = _FunctionProxy(function_name)
        self.unpack_output = unpack_output
        self.unwrap = True
        self.pipeline_thunks: tuple = ()

        # Same algorithm as Thunk.__init__ (core.py lines 78-81)
        string_repr = f"{source_hash}{STRING_REPR_DELIMITER}{unpack_output}"
        self.hash: str = sha256(string_repr.encode()).hexdigest()
        self.generates_file = False


class MatlabPipelineThunk:
    """Proxy for a specific MATLAB function invocation.

    Satisfies the same duck-typing contract as ``thunk.core.PipelineThunk``.
    Reuses ``classify_inputs`` and the lineage-hash algorithm from thunk-lib
    so that cache lookups and lineage extraction work unchanged.

    Parameters
    ----------
    matlab_thunk : MatlabThunk
        The parent thunk (function identity).
    inputs : dict
        Mapping of argument names (``"arg_0"``, ``"arg_1"``, ...) to
        Python-side values (BaseVariable instances, ThunkOutputs, or
        plain scalars/arrays).
    """

    def __init__(self, matlab_thunk: MatlabThunk, inputs: dict):
        self.thunk = matlab_thunk
        self.inputs: dict = dict(inputs)
        self.outputs: tuple = ()
        self.unwrap = True

    def compute_lineage_hash(self) -> str:
        """Compute lineage hash — identical algorithm to PipelineThunk."""
        classified = classify_inputs(self.inputs)
        input_tuples = [c.to_cache_tuple() for c in classified]
        hash_input = f"{self.thunk.hash}{STRING_REPR_DELIMITER}{input_tuples}"
        return sha256(hash_input.encode()).hexdigest()

    @property
    def hash(self) -> str:
        return self.compute_lineage_hash()


# ---------------------------------------------------------------------------
# Helper functions called from MATLAB
# ---------------------------------------------------------------------------


def check_cache(pipeline_thunk: MatlabPipelineThunk):
    """Check if a computation is already cached.

    Returns
    -------
    list or None
        List of cached output values (raw data), or None on miss.
    """
    from thunk.core import Thunk

    if Thunk.query is not None:
        try:
            return Thunk.query.find_by_lineage(pipeline_thunk)
        except Exception:
            pass
    return None


def make_thunk_output(pipeline_thunk: MatlabPipelineThunk, output_num: int, data):
    """Create a real ThunkOutput backed by a MatlabPipelineThunk.

    The returned object is a genuine ``thunk.core.ThunkOutput`` instance,
    so ``isinstance`` checks in ``save_variable`` pass.
    """
    from thunk.core import ThunkOutput

    return ThunkOutput(pipeline_thunk, output_num, True, data)


def register_matlab_variable(type_name: str, schema_version: int = 1):
    """Create a Python surrogate BaseVariable subclass for a MATLAB type.

    The surrogate is auto-registered in ``BaseVariable._all_subclasses``
    via ``__init_subclass__`` and, if a database is configured, registered
    with the ``DatabaseManager`` as well.

    Returns the surrogate class.
    """
    from scidb.variable import BaseVariable

    existing = BaseVariable.get_subclass_by_name(type_name)
    if existing is not None:
        return existing

    surrogate = type(type_name, (BaseVariable,), {"schema_version": schema_version})

    try:
        from scidb.database import get_database
        get_database().register(surrogate)
    except Exception:
        pass  # Database not yet configured; will register on configure_database

    return surrogate


def save_batch_bridge(type_name, data_values, metadata_keys, metadata_columns, common_metadata=None, db=None):
    """Bridge function for MATLAB save_from_table.

    Accepts columnar data (one list per column) from MATLAB and assembles
    the (data_value, metadata_dict) tuples that DatabaseManager.save_batch()
    expects.  This avoids per-row MATLAB↔Python round-trips.

    Parameters
    ----------
    type_name : str
        Variable class name (e.g. "StepLength").
    data_values : list or numpy array
        One data value per row.
    metadata_keys : list of str
        Metadata column names, same order as metadata_columns.
    metadata_columns : list of (list or numpy array)
        One inner list/array per metadata key, each with one value per row.
    common_metadata : dict or None
        Extra metadata applied to every row.
    db : DatabaseManager or None
        Optional database; uses global default when None.

    Returns
    -------
    list of str
        Record IDs for each saved row.
    """
    from scidb.variable import BaseVariable
    from scidb.database import get_database

    cls = BaseVariable.get_subclass_by_name(type_name)
    if cls is None:
        raise ValueError(
            f"Variable type '{type_name}' is not registered. "
            f"Call scidb.register_variable('{type_name}') first."
        )

    _db = db if db is not None and not isinstance(db, type(None)) else get_database()
    common = dict(common_metadata) if common_metadata else {}
    keys = list(metadata_keys)

    # Bulk-convert numpy arrays to native Python lists (one call instead of
    # N per-element .item() calls).  Plain Python lists pass through unchanged.
    if hasattr(data_values, 'tolist'):
        data_list = data_values.tolist()
    else:
        data_list = [v.item() if hasattr(v, 'item') else v for v in data_values]

    meta_lists = []
    for j in range(len(keys)):
        col = metadata_columns[j]
        if hasattr(col, 'tolist'):
            meta_lists.append(col.tolist())
        else:
            meta_lists.append([v.item() if hasattr(v, 'item') else v for v in col])

    n = len(data_list)
    data_items = []
    for i in range(n):
        meta = dict(common)
        for j, key in enumerate(keys):
            meta[key] = meta_lists[j][i]
        data_items.append((data_list[i], meta))

    return _db.save_batch(cls, data_items)


def get_surrogate_class(type_name: str):
    """Retrieve the Python surrogate class for a MATLAB variable type.

    Raises ValueError if not registered.
    """
    from scidb.variable import BaseVariable

    cls = BaseVariable.get_subclass_by_name(type_name)
    if cls is None:
        raise ValueError(
            f"MATLAB variable type '{type_name}' is not registered. "
            f"Call scidb.register_variable('{type_name}') first."
        )
    return cls
