function py_class = ensure_registered(type_name)
%ENSURE_REGISTERED  Auto-register a MATLAB variable type with Python.
%
%   PY_CLASS = scidb.internal.ensure_registered(TYPE_NAME) ensures that
%   the given type name has a Python surrogate BaseVariable subclass.
%   Returns the surrogate class for use in database calls.
%
%   This is called automatically by BaseVariable instance methods (save,
%   load, etc.) — users do not need to call this directly.

    py_class = py.scidb_matlab.bridge.register_matlab_variable(type_name);
end
