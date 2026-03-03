function c = pylist_to_cell(py_list)
%PYLIST_TO_CELL  Convert a Python list to a MATLAB cell array.
%
%   c = scidb.internal.pylist_to_cell(py.list(...))
%
%   Each element is recursively converted.  Python dicts become structs,
%   Python strings become MATLAB strings, etc.

    if isa(py_list, 'py.NoneType')
        c = {};
        return;
    end

    raw = cell(py_list);
    c = cell(1, numel(raw));
    for i = 1:numel(raw)
        item = raw{i};
        if isa(item, 'py.dict')
            c{i} = scidb.internal.pydict_to_struct(item);
        else
            c{i} = scidb.internal.from_python(item);
        end
    end
end
