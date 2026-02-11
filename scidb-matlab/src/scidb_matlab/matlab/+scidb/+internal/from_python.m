function data = from_python(py_obj)
%FROM_PYTHON  Convert a Python object back to a native MATLAB type.
%
%   Handles numpy ndarrays, Python scalars (float, int, str, bool),
%   lists, and dicts.

    if isa(py_obj, 'py.NoneType')
        data = [];

    elseif isa(py_obj, 'py.numpy.ndarray')
        % Ensure C-contiguous before extracting
        py_c = py.numpy.ascontiguousarray(py_obj);
        data = double(py_c);

    elseif isa(py_obj, 'py.float')
        data = double(py_obj);

    elseif isa(py_obj, 'py.int')
        data = double(py_obj);

    elseif isa(py_obj, 'py.str')
        data = string(py_obj);

    elseif isa(py_obj, 'py.bool')
        data = logical(py_obj);

    elseif isa(py_obj, 'py.list')
        c = cell(py_obj);
        data = cell(1, numel(c));
        for i = 1:numel(c)
            data{i} = scidb.internal.from_python(c{i});
        end

    elseif isa(py_obj, 'py.pandas.core.frame.DataFrame')
        % pandas DataFrame -> MATLAB table
        col_names = cell(py_obj.columns.tolist());
        args = cell(1, numel(col_names));
        for i = 1:numel(col_names)
            col = py_obj{col_names{i}};
            args{i} = scidb.internal.from_python(col.to_numpy());
            % Ensure column vector
            args{i} = args{i}(:);
        end
        col_name_strs = cellfun(@string, col_names, 'UniformOutput', false);
        data = table(args{:}, 'VariableNames', [col_name_strs{:}]);

    elseif isa(py_obj, 'py.dict')
        data = scidb.internal.pydict_to_struct(py_obj);

    else
        % Last resort: try MATLAB's automatic conversion
        try
            data = double(py_obj);
        catch
            data = py_obj;  % Return raw Python object
        end
    end
end
