function var = load(type_token, varargin)
%SCIDB.LOAD  Load a variable from the database.
%
%   VAR = scidb.load(TypeClass, Name, Value, ...)
%   loads the latest version matching the given metadata.
%
%   Returns a scidb.BaseVariable with properties:
%     .data        - The loaded data (MATLAB native type)
%     .record_id   - Unique record identifier
%     .metadata    - Struct of metadata key-value pairs
%     .py_obj      - Python BaseVariable (used internally by Thunk)
%
%   Arguments:
%       type_token - A BaseVariable subclass instance (e.g. RawSignal)
%
%   Name-Value Arguments:
%       Any metadata key-value pairs (e.g. subject=1, session="A")
%       version - Specific record_id to load (default "latest")
%
%   Example:
%       raw = scidb.load(RawSignal, subject=1, session="A");
%       disp(raw.data);

    type_name = scidb.internal.resolve_type_name(type_token);

    % Separate 'version' from metadata
    [metadata_args, version] = scidb.internal.split_version_arg(varargin{:});

    % Get the Python surrogate class
    py_class = py.scidb_matlab.bridge.get_surrogate_class(type_name);

    % Build Python metadata dict
    py_metadata = scidb.internal.metadata_to_pydict(metadata_args{:});

    % Call Python's load
    py_db = py.scidb.database.get_database();
    py_var = py_db.load(py_class, py_metadata, version=char(version));

    % Convert to MATLAB BaseVariable
    var = scidb.BaseVariable();
    var.data = scidb.internal.from_python(py_var.data);
    var.record_id = string(py_var._record_id);
    var.content_hash = string(py_var._content_hash);

    % lineage_hash may be Python None
    py_lh = py_var._lineage_hash;
    if ~isa(py_lh, 'py.NoneType')
        var.lineage_hash = string(py_lh);
    end

    % Convert metadata
    py_meta = py_var._metadata;
    if ~isa(py_meta, 'py.NoneType')
        var.metadata = scidb.internal.pydict_to_struct(py_meta);
    end

    % Keep Python shadow for thunk input classification
    var.py_obj = py_var;
end
