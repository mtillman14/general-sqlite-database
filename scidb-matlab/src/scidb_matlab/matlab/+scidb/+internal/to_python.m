function py_obj = to_python(data)
%TO_PYTHON  Convert MATLAB data to a Python object for database storage.
%
%   Handles: double/single/integer arrays, scalars, strings, logicals.
%   Arrays are converted to C-contiguous numpy ndarrays so that
%   canonical_hash produces consistent results.

    if isstring(data) && isscalar(data)
        py_obj = char(data);

    elseif ischar(data)
        py_obj = py.str(data);

    elseif islogical(data) && isscalar(data)
        py_obj = py.bool(data);

    elseif isnumeric(data) && isscalar(data)
        if isfloat(data)
            py_obj = py.float(double(data));
        else
            py_obj = py.int(int64(data));
        end

    elseif isnumeric(data)
        % Multi-element array -> numpy ndarray (C-contiguous / row-major)
        %
        % MATLAB is column-major; numpy default is row-major.  We
        % transpose so that the logical element order matches, then
        % request C-contiguous layout for deterministic hashing.
        if ismatrix(data) && ~isvector(data)
            % 2-D matrix: transpose so row-major matches MATLAB convention
            flat = data';
        else
            flat = data;
        end

        % Determine numpy dtype string
        dtype = matlab_dtype_to_numpy(data);

        py_flat = py.numpy.array(flat(:)', pyargs('dtype', dtype));
        py_shape = py.tuple(int64(size(data)));
        py_obj = py_flat.reshape(py_shape, pyargs('order', 'C'));
        py_obj = py.numpy.ascontiguousarray(py_obj);

    elseif iscell(data)
        % Cell array -> Python list
        py_obj = py.list();
        for i = 1:numel(data)
            py_obj.append(scidb.internal.to_python(data{i}));
        end

    elseif istable(data)
        % MATLAB table -> pandas DataFrame
        col_names = data.Properties.VariableNames;
        py_dict = py.dict();
        for i = 1:numel(col_names)
            col = data.(col_names{i});
            if iscategorical(col)
                col = string(col);
            end
            py_dict{col_names{i}} = scidb.internal.to_python(col);
        end
        py_obj = py.pandas.DataFrame(py_dict);

    elseif isstruct(data) && isscalar(data)
        % Scalar struct -> Python dict
        py_obj = py.dict();
        fns = fieldnames(data);
        for i = 1:numel(fns)
            py_obj{fns{i}} = scidb.internal.to_python(data.(fns{i}));
        end

    else
        error('scidb:UnsupportedType', ...
            'Cannot convert MATLAB type "%s" to Python.', class(data));
    end
end


function dtype = matlab_dtype_to_numpy(data)
%MATLAB_DTYPE_TO_NUMPY  Map MATLAB numeric class to numpy dtype string.
    switch class(data)
        case 'double',  dtype = 'float64';
        case 'single',  dtype = 'float32';
        case 'int8',    dtype = 'int8';
        case 'int16',   dtype = 'int16';
        case 'int32',   dtype = 'int32';
        case 'int64',   dtype = 'int64';
        case 'uint8',   dtype = 'uint8';
        case 'uint16',  dtype = 'uint16';
        case 'uint32',  dtype = 'uint32';
        case 'uint64',  dtype = 'uint64';
        otherwise
            dtype = 'float64';
    end
end
