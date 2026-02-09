function prov = provenance(type_token, varargin)
%SCIDB.PROVENANCE  Get the provenance (lineage) of a variable.
%
%   PROV = scidb.provenance(TypeClass, Name, Value, ...)
%   returns a struct with fields: function_name, function_hash, inputs,
%   constants.  Returns empty [] if no lineage is recorded.
%
%   Arguments:
%       type_token - A BaseVariable subclass instance (e.g. RawSignal)
%
%   Name-Value Arguments:
%       Metadata key-value pairs identifying the variable
%       version - Specific record_id (optional)
%
%   Example:
%       p = scidb.provenance(ProcessedSignal, subject=1, session="A");
%       fprintf("Computed by: %s\n", p.function_name);

    type_name = scidb.internal.resolve_type_name(type_token);

    [metadata_args, version] = scidb.internal.split_version_arg(varargin{:});
    py_class = py.scidb_matlab.bridge.get_surrogate_class(type_name);
    py_kwargs = scidb.internal.metadata_to_pykwargs(metadata_args{:});

    py_db = py.scidb.database.get_database();
    if version ~= "latest"
        py_result = py_db.get_provenance(py_class, version=char(version), py_kwargs{:});
    else
        py_result = py_db.get_provenance(py_class, py_kwargs{:});
    end

    % Convert Python dict or None
    if isa(py_result, 'py.NoneType')
        prov = [];
    else
        prov.function_name = string(py_result{'function_name'});
        prov.function_hash = string(py_result{'function_hash'});
        prov.inputs        = scidb.internal.pylist_to_cell(py_result{'inputs'});
        prov.constants     = scidb.internal.pylist_to_cell(py_result{'constants'});
    end
end
