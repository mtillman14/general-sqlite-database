function vars = load_all(type_token, varargin)
%SCIDB.LOAD_ALL  Load all variables matching the given metadata.
%
%   VARS = scidb.load_all(TypeClass, Name, Value, ...)
%   returns an array of scidb.BaseVariable objects.
%
%   Arguments:
%       type_token - A BaseVariable subclass instance (e.g. RawSignal)
%
%   Name-Value Arguments:
%       Any metadata key-value pairs for filtering
%
%   Example:
%       all_signals = scidb.load_all(RawSignal, subject=1);
%       for i = 1:numel(all_signals)
%           disp(all_signals(i).metadata);
%       end

    type_name = scidb.internal.resolve_type_name(type_token);

    py_class = py.scidb_matlab.bridge.get_surrogate_class(type_name);
    py_metadata = scidb.internal.metadata_to_pydict(varargin{:});

    py_db = py.scidb.database.get_database();
    py_gen = py_db.load_all(py_class, py_metadata);

    % Collect generator results into a MATLAB array
    vars = scidb.BaseVariable.empty();
    py_iter = py.builtins.iter(py_gen);

    while true
        try
            py_var = py.builtins.next(py_iter);
        catch
            break;  % StopIteration
        end

        v = scidb.BaseVariable();
        v.data = scidb.internal.from_python(py_var.data);
        v.record_id = string(py_var._record_id);
        v.content_hash = string(py_var._content_hash);

        py_lh = py_var._lineage_hash;
        if ~isa(py_lh, 'py.NoneType')
            v.lineage_hash = string(py_lh);
        end

        py_meta = py_var._metadata;
        if ~isa(py_meta, 'py.NoneType')
            v.metadata = scidb.internal.pydict_to_struct(py_meta);
        end

        v.py_obj = py_var;
        vars(end+1) = v; %#ok<AGROW>
    end
end
