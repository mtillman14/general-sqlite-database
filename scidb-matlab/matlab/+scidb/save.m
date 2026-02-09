function record_id = save(type_token, data, varargin)
%SCIDB.SAVE  Save data to the database.
%
%   RECORD_ID = scidb.save(TypeClass, DATA, Name, Value, ...)
%   saves DATA under the given variable type with the specified metadata.
%
%   The first argument is a BaseVariable subclass used as a type token.
%   MATLAB constructs a lightweight instance automatically when you pass
%   the class name directly.
%
%   DATA can be:
%     - A numeric array (double, single, integer types)
%     - A scalar (double, int, string, logical)
%     - A scidb.ThunkOutput from a thunked computation (lineage is
%       automatically extracted and stored)
%     - A scidb.BaseVariable instance (re-save with new metadata)
%
%   Arguments:
%       type_token - A BaseVariable subclass instance (e.g. RawSignal)
%       data       - The data to save
%
%   Name-Value Arguments:
%       Any metadata key-value pairs (e.g. subject=1, session="A")
%
%   Returns:
%       record_id - The unique record ID of the saved data (char)
%
%   Example:
%       scidb.save(RawSignal, randn(100,3), subject=1, session="A");
%
%       result = my_thunk(input_var, 2.5);
%       scidb.save(Processed, result, subject=1, session="A");

    type_name = scidb.internal.resolve_type_name(type_token);

    % Get the Python surrogate class
    py_class = py.scidb_matlab.bridge.get_surrogate_class(type_name);

    % Marshal data to Python
    if isa(data, 'scidb.ThunkOutput')
        % Pass the real Python ThunkOutput — isinstance checks pass,
        % extract_lineage works, lineage is saved automatically.
        py_data = data.py_obj;
    elseif isa(data, 'scidb.BaseVariable')
        % Re-saving a loaded variable — pass the Python shadow.
        py_data = data.py_obj;
    else
        % Raw MATLAB data — convert to Python/numpy.
        py_data = scidb.internal.to_python(data);
    end

    % Build Python kwargs from name-value pairs
    py_kwargs = scidb.internal.metadata_to_pykwargs(varargin{:});

    % Call Python's save_variable
    py_db = py.scidb.database.get_database();
    py_record_id = py_db.save_variable(py_class, py_data, py_kwargs{:});
    record_id = char(py_record_id);
end
