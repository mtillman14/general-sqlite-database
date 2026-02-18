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
        dtype_kind = string(py_c.dtype.kind);
        if dtype_kind == "b"
            % bool array -> logical
            data = logical(double(py_c));
        elseif dtype_kind == "O"
            % Object array -> cell array, convert each element
            py_list = py_c.tolist();
            c = cell(py_list);
            data = cell(1, numel(c));
            for i = 1:numel(c)
                data{i} = scidb.internal.from_python(c{i});
            end
        else
            data = double(py_c);
        end

    elseif isa(py_obj, 'py.float')
        data = double(py_obj);

    elseif isa(py_obj, 'py.int')
        data = double(py_obj);

    elseif isa(py_obj, 'py.str')
        data = string(py_obj);

    elseif isa(py_obj, 'py.bool')
        data = logical(py_obj);

    elseif isa(py_obj, 'py.datetime.datetime')
        data = datetime(string(py_obj.isoformat()), 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');

    elseif isa(py_obj, 'py.list')
        c = cell(py_obj);
        n = numel(c);
        data = cell(1, n);
        all_str = n > 0;
        for i = 1:n
            data{i} = scidb.internal.from_python(c{i});
            if all_str && ~isstring(data{i})
                all_str = false;
            end
        end
        if all_str
            % All-string list -> string array (round-trips string arrays)
            data = [data{:}];
        end

    elseif isa(py_obj, 'py.pandas.core.frame.DataFrame')
        % pandas DataFrame -> MATLAB table
        col_names = cell(py_obj.columns.tolist());
        args = cell(1, numel(col_names));
        for i = 1:numel(col_names)
            col = py_obj{col_names{i}};
            dtype_str = string(col.dtype.name);
            if startsWith(dtype_str, "datetime")
                % datetime64 column -> MATLAB datetime via ISO strings
                iso_strs = cell(col.dt.strftime('%Y-%m-%dT%H:%M:%S.%f').tolist());
                args{i} = datetime(iso_strs, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS');
            elseif dtype_str == "object"
                % Object column (e.g. dicts/structs) -> cell array via from_python
                py_list = col.tolist();
                c = cell(py_list);
                col_data = cell(numel(c), 1);
                for k = 1:numel(c)
                    col_data{k} = scidb.internal.from_python(c{k});
                end
                % Reconstruct matrix if all elements are same-size numeric
                % (round-trips multi-column table variables like Nx2 matrices)
                args{i} = try_stack_numeric(col_data);
            else
                args{i} = scidb.internal.from_python(col.to_numpy());
            end
            % Ensure column vector (preserve multi-column matrices)
            if isvector(args{i})
                args{i} = args{i}(:);
            end
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


function data = try_stack_numeric(data)
%TRY_STACK_NUMERIC  Stack a cell of same-size numeric vectors into a matrix.
%   If every element is numeric with identical size, vertcat them into a
%   matrix (round-trips multi-column table variables).  Otherwise return
%   the cell array unchanged.
    if isempty(data) || ~isnumeric(data{1}), return; end
    ref_sz = size(data{1});
    for k = 2:numel(data)
        if ~isnumeric(data{k}) || ~isequal(size(data{k}), ref_sz)
            return;
        end
    end
    data = vertcat(data{:});
end
