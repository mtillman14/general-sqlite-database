function py_dict = metadata_to_pydict(varargin)
%METADATA_TO_PYDICT  Convert MATLAB name-value pairs to a Python dict.
%
%   py_dict = scidb.internal.metadata_to_pydict('subject', 1, 'session', 'A')
%   returns py.dict({"subject": 1, "session": "A"})

    if mod(numel(varargin), 2) ~= 0
        error('scidb:InvalidMetadata', ...
            'Metadata must be specified as name-value pairs.');
    end

    py_dict = py.dict();
    for i = 1:2:numel(varargin)
        key = varargin{i};
        val = varargin{i+1};

        if isstring(key)
            key = char(key);
        end

        % Convert MATLAB values to Python-compatible types
        if isstring(val) && isscalar(val)
            py_dict{key} = char(val);
        elseif isstring(val) && ~isscalar(val)
            % String array (non-scalar) → Python list for "match any"
            py_dict{key} = py.list(cellfun(@char, num2cell(val), 'UniformOutput', false));
        elseif isnumeric(val) && isscalar(val)
            % Whole numbers → py.int so str() gives "1" not "1.0"
            % (schema keys are VARCHAR in DuckDB)
            if val == floor(val)
                py_dict{key} = py.int(int64(val));
            else
                py_dict{key} = py.float(double(val));
            end
        elseif isnumeric(val) && ~isscalar(val)
            % Numeric array (non-scalar) → Python list for "match any"
            % Whole numbers → py.int so str() gives "1" not "1.0"
            if all(val == floor(val))
                py_dict{key} = py.list(arrayfun(@(x) py.int(int64(x)), val, 'UniformOutput', false));
            else
                py_dict{key} = py.list(num2cell(val));
            end
        elseif ischar(val)
            py_dict{key} = py.str(val);
        else
            py_dict{key} = val;
        end
    end
end
