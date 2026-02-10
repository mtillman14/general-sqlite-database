function s = pydict_to_struct(py_dict)
%PYDICT_TO_STRUCT  Convert a Python dict to a MATLAB struct.
%
%   s = scidb.internal.pydict_to_struct(py.dict(...))
%
%   Keys must be strings.  Values are recursively converted via
%   from_python().

    if isa(py_dict, 'py.NoneType')
        s = struct();
        return;
    end

    s = struct();
    py_keys = py.builtins.list(py_dict.keys());
    n = int64(py.builtins.len(py_keys));

    for i = 1:n
        key = char(py_keys{i});
        val = py_dict{key};

        % Sanitise key for MATLAB field name (replace invalid chars)
        field = matlab.lang.makeValidName(key);
        s.(field) = scidb.internal.from_python(val);
    end
end
