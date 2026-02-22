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
        all_numeric_scalar = n > 0;
        all_logical_scalar = n > 0;
        for i = 1:n
            data{i} = scidb.internal.from_python(c{i});
            if all_str && ~isstring(data{i})
                all_str = false;
            end
            if all_numeric_scalar && ~(isnumeric(data{i}) && isscalar(data{i}))
                all_numeric_scalar = false;
            end
            if all_logical_scalar && ~(islogical(data{i}) && isscalar(data{i}))
                all_logical_scalar = false;
            end
        end
        if all_str
            % All-string list -> string array (round-trips string arrays)
            data = [data{:}];
        elseif all_numeric_scalar
            % All-scalar-numeric list -> numeric vector
            data = [data{:}];
        elseif all_logical_scalar
            % All-scalar-logical list -> logical vector
            data = [data{:}];
        end

    elseif isa(py_obj, 'py.pandas.core.frame.DataFrame') || isa(py_obj, 'py.pandas.DataFrame')
        data = convert_dataframe(py_obj);

    elseif isa(py_obj, 'py.dict')
        data = scidb.internal.pydict_to_struct(py_obj);

    else
        % Fallback: isa() can miss pandas DataFrames depending on MATLAB
        % version / class proxy resolution.  Use Python isinstance as a
        % robust secondary check before giving up.
        is_df = false;
        try
            is_df = logical(py.builtins.isinstance(py_obj, py.pandas.DataFrame));
        catch
        end

        if is_df
            data = convert_dataframe(py_obj);
        else
            % Last resort: try MATLAB's automatic conversion
            try
                data = double(py_obj);
            catch
                data = py_obj;  % Return raw Python object
            end
        end
    end
end


function data = convert_dataframe(py_obj)
%CONVERT_DATAFRAME  Convert a pandas DataFrame to a MATLAB table.
    py_cols = py_obj.columns.tolist();
    col_names = cell(py_cols);
    args = cell(1, numel(col_names));
    n_rows = int64(py.builtins.len(py_obj));
    % fprintf('[DIAG convert_dataframe] %d columns, %d rows\n', numel(col_names), n_rows);
    for i = 1:numel(col_names)
        col_key = col_names{i};
        col = py.operator.getitem(py_obj, col_key);
        dtype_str = string(col.dtype.name);
        % fprintf('[DIAG convert_dataframe] col %d/%d "%s": pandas dtype=%s\n', ...
            % i, numel(col_names), string(col_key), dtype_str);
        if startsWith(dtype_str, "datetime")
            % datetime64 column -> MATLAB datetime via ISO strings
            iso_strs = cell(col.dt.strftime('%Y-%m-%dT%H:%M:%S.%f').tolist());
            args{i} = datetime(iso_strs, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS');
            % fprintf('[DIAG convert_dataframe]   -> datetime [%s]\n', ...
                % num2str(size(args{i})));
        elseif dtype_str == "object"
            % Object column (e.g. dicts/structs) -> cell array via from_python
            py_list = col.tolist();
            c = cell(py_list);
            col_data = cell(numel(c), 1);
            for k = 1:numel(c)
                py_elem = c{k};
                if k == 1
                    % fprintf('[DIAG convert_dataframe]   object col, first elem Python type=%s\n', ...
                        % string(py.builtins.getattr(py.type(py_elem), '__name__')));
                end
                col_data{k} = scidb.internal.from_python(py_elem);
                if k == 1
                    % fprintf('[DIAG convert_dataframe]   after from_python: MATLAB class=%s, size=[%s]\n', ...
                        % class(col_data{k}), num2str(size(col_data{k})));
                    if isstring(col_data{k})
                        % fprintf('[DIAG convert_dataframe]   STRING VALUE: "%s"\n', ...
                            % extractBefore(col_data{k}, min(120, strlength(col_data{k})+1)));
                    end
                end
                % Parse stringified arrays (e.g. "[[false], [true], ...]")
                % that result from nested-list storage in DuckDB.
                if isstring(col_data{k}) && strlength(col_data{k}) > 1 ...
                        && startsWith(col_data{k}, "[")
                    try
                        col_data{k} = jsondecode(char(col_data{k}));
                        if k == 1
                            % fprintf('[DIAG convert_dataframe]   jsondecode succeeded: class=%s, size=[%s]\n', ...
                                % class(col_data{k}), num2str(size(col_data{k})));
                        end
                    catch ME
                        if k == 1
                            % fprintf('[DIAG convert_dataframe]   jsondecode FAILED: %s\n', ME.message);
                        end
                    end
                end
            end
            % DIAGNOSTIC: Log pre-stacking state
            % fprintf('[DIAG convert_dataframe]   pre-stack: %d cells\n', numel(col_data));
            if numel(col_data) > 0
                % fprintf('[DIAG convert_dataframe]   cell{1} class=%s, size=[%s]\n', ...
                    % class(col_data{1}), num2str(size(col_data{1})));
                if isstruct(col_data{1})
                    % fprintf('[DIAG convert_dataframe]   struct fields: %s\n', ...
                        % strjoin(string(fieldnames(col_data{1})), ', '));
                    % Check each field type
                    fns = fieldnames(col_data{1});
                    for fi = 1:numel(fns)
                        fval = col_data{1}.(fns{fi});
                        % fprintf('[DIAG convert_dataframe]     field "%s": class=%s, size=[%s]\n', ...
                           % fns{fi}, class(fval), num2str(size(fval)));
                        if isstruct(fval)
                            subfns = fieldnames(fval);
                            % fprintf('[DIAG convert_dataframe]       sub-struct fields: %s\n', ...
                                % strjoin(string(subfns), ', '));
                            for sfi = 1:min(3, numel(subfns))
                                sfval = fval.(subfns{sfi});
                                % fprintf('[DIAG convert_dataframe]         "%s": class=%s, size=[%s]\n', ...
                                    % subfns{sfi}, class(sfval), num2str(size(sfval)));
                            end
                        end
                    end
                end
            end
            % Reconstruct matrix if all elements are same-size numeric
            % (round-trips multi-column table variables like Nx2 matrices)
            col_data = try_stack_numeric(col_data);
            % Coalesce all-string cell arrays into a MATLAB string array
            % (round-trips string columns that pandas stores as object dtype)
            col_data = try_stack_strings(col_data);
            % Convert cell of structs to struct array so table columns
            % are accessible as t.field.subfield instead of t.field{1}.subfield
            col_data_before_struct_stack = col_data;
            args{i} = try_stack_structs(col_data);
            if iscell(col_data_before_struct_stack) && isstruct(col_data_before_struct_stack{1}) && iscell(args{i})
                % fprintf('[DIAG convert_dataframe]   try_stack_structs FAILED (still cell). ');
                % Diagnose why: check field name consistency
                for si = 1:min(3, numel(col_data_before_struct_stack))
                    % fprintf('row %d fields: %s; ', si, ...
                        % strjoin(string(sort(fieldnames(col_data_before_struct_stack{si}))), ','));
                end
                % fprintf('\n');
            elseif isstruct(args{i})
                % fprintf('[DIAG convert_dataframe]   try_stack_structs OK -> struct array [%s]\n', ...
                    % num2str(size(args{i})));
            end
        else
            args{i} = scidb.internal.from_python(col.to_numpy());
            % fprintf('[DIAG convert_dataframe]   -> %s [%s]\n', ...
                % class(args{i}), num2str(size(args{i})));
            % pandas 3.0+ returns StringDtype for text columns; from_python
            % converts these to cell arrays.  Stack into string arrays.
            if iscell(args{i})
                args{i} = try_stack_strings(args{i});
            end
        end
        % Ensure column vector — but only when the number of elements                                                                                                                
        % matches the DataFrame row count.  Otherwise a 1-row DataFrame                                                                                                              
        % with a 14-element array value would be reshaped from 1x14 to                                                                                                               
        % 14x1, making the table think there are 14 rows.
        if isvector(args{i}) && numel(args{i}) == n_rows
            args{i} = args{i}(:);
        end
        % fprintf('[DIAG convert_dataframe]   FINAL: class=%s, size=[%s]\n', ...
            % class(args{i}), num2str(size(args{i})));
    end
    col_name_strs = cellfun(@string, col_names, 'UniformOutput', false);
    data = table;
    for i = 1:numel(args)
        if numel(args{i}) == n_rows
            % Per-row values (column vector length matches row count): assign directly
            data.(col_name_strs{i}) = args{i};
        else
            % Per-row array values (e.g. time series stored in one cell per row):
            % cell-wrap so the table sees one cell per row, not one row per element.
            data.(col_name_strs{i}) = args(i);
        end
    end
end


function data = try_stack_numeric(data)
%TRY_STACK_NUMERIC  Stack a cell of same-size numeric vectors into a matrix.
%   If every element is numeric with identical size, vertcat them into a
%   matrix (round-trips multi-column table variables).  Otherwise return
%   the cell array unchanged.
    if ~iscell(data) || isempty(data) || ~isnumeric(data{1}), return; end
    ref_sz = size(data{1});
    for k = 2:numel(data)
        if ~isnumeric(data{k}) || ~isequal(size(data{k}), ref_sz)
            return;
        end
    end
    data = vertcat(data{:});
end


function data = try_stack_strings(data)
%TRY_STACK_STRINGS  Convert a cell array of scalar strings to a string array.
%   If every element is a scalar MATLAB string, concatenate into a column
%   string vector (round-trips string columns stored as pandas object dtype).
%   Otherwise return the cell array unchanged.
    if ~iscell(data) || isempty(data) || ~isstring(data{1}) || ~isscalar(data{1}), return; end
    for k = 2:numel(data)
        if ~isstring(data{k}) || ~isscalar(data{k})
            return;
        end
    end
    data = vertcat(data{:});
end


function data = try_stack_structs(data)
%TRY_STACK_STRUCTS  Convert a cell array of structs to a struct array.
%   If every element is a struct with identical fields, vertcat them into
%   a struct array.  This allows table access as t.field.subfield instead
%   of t.field{1}.subfield.  Otherwise return the data unchanged.
    if ~iscell(data) || isempty(data) || ~isstruct(data{1}), return; end
    ref_fields = sort(fieldnames(data{1}));
    for k = 2:numel(data)
        if ~isstruct(data{k})
            % fprintf('[DIAG try_stack_structs] FAIL at row %d: not a struct (class=%s)\n', ...
                % k, class(data{k}));
            return;
        end
        cur_fields = sort(fieldnames(data{k}));
        if ~isequal(cur_fields, ref_fields)
            % fprintf('[DIAG try_stack_structs] FAIL at row %d: field mismatch\n', k);
            % fprintf('[DIAG try_stack_structs]   ref fields: %s\n', strjoin(string(ref_fields), ', '));
            % fprintf('[DIAG try_stack_structs]   row %d fields: %s\n', k, strjoin(string(cur_fields), ', '));
            return;
        end
    end
    % Check if vertcat will work (fields must have compatible sizes)
    try
        data = vertcat(data{:});
    catch ME
        % fprintf('[DIAG try_stack_structs] vertcat FAILED: %s\n', ME.message);
        % Diagnose: check field value sizes across rows
        for fi = 1:numel(ref_fields)
            sizes = cell(numel(data), 1);
            for k = 1:min(3, numel(data))
                sizes{k} = num2str(size(data{k}.(ref_fields{fi})));
            end
            % fprintf('[DIAG try_stack_structs]   field "%s" sizes: %s\n', ...
                % ref_fields{fi}, strjoin(string(sizes(1:min(3,numel(data)))), ' | '));
        end
    end
end
