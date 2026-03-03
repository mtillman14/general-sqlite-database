function py_val = to_python_input(arg)
%TO_PYTHON_INPUT  Marshal a MATLAB thunk argument to a Python object.
%
%   For scidb.ThunkOutput and scidb.BaseVariable, returns the Python
%   shadow (.py_obj) so that classify_inputs() in thunk-lib sees the
%   real Python type and input classification works unchanged.
%
%   For raw MATLAB data (scalars, arrays), converts to the Python
%   equivalent via to_python().

    if isa(arg, 'scidb.ThunkOutput')
        % Pass the real Python ThunkOutput
        py_val = arg.py_obj;

    elseif isa(arg, 'scidb.BaseVariable')
        % Pass the real Python BaseVariable (with _record_id, _lineage_hash)
        py_val = arg.py_obj;

    else
        % Convert raw MATLAB data
        py_val = scidb.internal.to_python(arg);
    end
end
