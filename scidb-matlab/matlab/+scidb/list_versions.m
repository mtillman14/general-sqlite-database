function versions = list_versions(type_token, varargin)
%SCIDB.LIST_VERSIONS  List all versions at a schema location.
%
%   VERSIONS = scidb.list_versions(TypeClass, Name, Value, ...)
%   returns a struct array with fields: record_id, schema, version,
%   created_at.
%
%   Arguments:
%       type_token - A BaseVariable subclass instance (e.g. RawSignal)
%
%   Name-Value Arguments:
%       Schema metadata key-value pairs
%
%   Example:
%       v = scidb.list_versions(ProcessedSignal, subject=1, session="A");
%       for i = 1:numel(v)
%           fprintf("  %s  created %s\n", v(i).record_id, v(i).created_at);
%       end

    type_name = scidb.internal.resolve_type_name(type_token);

    py_class = py.scidb_matlab.bridge.get_surrogate_class(type_name);
    py_kwargs = scidb.internal.metadata_to_pykwargs(varargin{:});

    py_db = py.scidb.database.get_database();
    py_list = py_db.list_versions(py_class, py_kwargs{:});

    % Convert Python list of dicts to MATLAB struct array
    n = int64(py.builtins.len(py_list));
    versions = struct('record_id', {}, 'schema', {}, 'version', {}, 'created_at', {});

    for i = 1:n
        py_dict = py_list{i};
        versions(i).record_id  = string(py_dict{'record_id'});
        versions(i).schema     = scidb.internal.pydict_to_struct(py_dict{'schema'});
        versions(i).version    = scidb.internal.pydict_to_struct(py_dict{'version'});
        versions(i).created_at = string(py_dict{'created_at'});
    end
end
