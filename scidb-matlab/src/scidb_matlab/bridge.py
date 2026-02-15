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
        if isinstance(col, str):
            # Joined string from MATLAB (record-separator delimited)
            meta_lists.append(col.split('\x1e'))
        elif hasattr(col, 'tolist'):
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

    return "\n".join(_db.save_batch(cls, data_items))


# ---------------------------------------------------------------------------
# Batch cache — keeps data/py_vars in Python so they never cross to MATLAB's
# proxy layer.  MATLAB accesses individual items via get_batch_item().
# ---------------------------------------------------------------------------

_batch_cache = {}
_batch_id_counter = 0


def _cache_batch(data_list, py_vars_list):
    """Store data and py_vars lists server-side, return an integer handle."""
    global _batch_id_counter
    bid = _batch_id_counter
    _batch_id_counter += 1
    _batch_cache[bid] = (data_list, py_vars_list)
    return bid


def get_batch_item(batch_id, index):
    """Return (data, py_var) for one element from a cached batch."""
    data_list, py_vars_list = _batch_cache[int(batch_id)]
    i = int(index)
    return data_list[i], py_vars_list[i]


def free_batch(batch_id):
    """Release a cached batch."""
    _batch_cache.pop(int(batch_id), None)


def wrap_batch_bridge(py_vars_list):
    """Extract all fields from a list of BaseVariables into bulk format.

    Scalar fields are packed into newline-joined strings and metadata into
    a single JSON string.  The heavy ``data`` and ``py_vars`` lists are
    stored in a Python-side cache (never returned to MATLAB) and accessed
    one element at a time via ``get_batch_item(batch_id, index)``.

    Parameters
    ----------
    py_vars_list : list of BaseVariable
        Python BaseVariable instances to extract.

    Returns
    -------
    dict with keys:
        n              : int
        batch_id       : int  — handle for get_batch_item / free_batch
        record_ids     : str  — newline-joined
        content_hashes : str  — newline-joined
        lineage_hashes : str  — newline-joined ('' for None)
        version_ids    : str  — newline-joined int strings ('' for None)
        parameter_ids  : str  — newline-joined int strings ('' for None)
        json_meta      : str  — JSON array of metadata dicts
    """
    import json

    py_vars = list(py_vars_list) if not isinstance(py_vars_list, list) else py_vars_list
    n = len(py_vars)

    record_ids = []
    content_hashes = []
    lineage_hashes = []
    version_ids = []
    parameter_ids = []
    meta_dicts = []
    data = []

    for v in py_vars:
        record_ids.append(v.record_id or '')
        content_hashes.append(v.content_hash or '')
        lh = v.lineage_hash
        lineage_hashes.append(lh if lh is not None else '')
        vid = getattr(v, 'version_id', None)
        version_ids.append(str(vid) if vid is not None else '')
        pid = getattr(v, 'parameter_id', None)
        parameter_ids.append(str(pid) if pid is not None else '')
        meta = v.metadata
        meta_dicts.append(dict(meta) if meta is not None else {})
        data.append(v.data)

    batch_id = _cache_batch(data, py_vars)

    return {
        'n': n,
        'batch_id': batch_id,
        'record_ids': '\n'.join(record_ids),
        'content_hashes': '\n'.join(content_hashes),
        'lineage_hashes': '\n'.join(lineage_hashes),
        'version_ids': '\n'.join(version_ids),
        'parameter_ids': '\n'.join(parameter_ids),
        'json_meta': json.dumps(meta_dicts),
    }


def load_and_extract(py_class, metadata_dict, version_id='latest', db=None):
    """Load all matching variables and extract fields in bulk.

    Combines load_all -> list -> wrap_batch_bridge in one Python call.
    The intermediate BaseVariable list and data arrays stay in Python
    (accessed later via get_batch_item).  Only lightweight strings/JSON
    cross back to MATLAB.

    Parameters
    ----------
    py_class : type
        BaseVariable subclass to load.
    metadata_dict : dict
        Metadata filter (values can be lists for "match any").
    version_id : str or int
        Version filter ('latest', 'all', or an integer).
    db : DatabaseManager or None
        Optional database; uses global default when None.

    Returns
    -------
    dict
        Same format as wrap_batch_bridge (with batch_id, no data/py_vars).
    """
    import time
    from scidb.database import get_database

    _db = db if db is not None and not isinstance(db, type(None)) else get_database()

    t0 = time.time()
    gen = _db.load_all(py_class, dict(metadata_dict), version_id=version_id)
    py_vars = list(gen)  # materializes entirely in Python
    t1 = time.time()
    result = wrap_batch_bridge(py_vars)
    t2 = time.time()

    print(f"[load_and_extract] query+materialize: {t1-t0:.2f}s, "
          f"extract: {t2-t1:.2f}s, n={len(py_vars)}")
    return result


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
