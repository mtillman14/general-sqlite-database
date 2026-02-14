function [metadata_args, py_version_id] = split_load_all_args(varargin)
%SPLIT_LOAD_ALL_ARGS  Separate 'version_id' from other name-value metadata.
%
%   [metadata_args, py_version_id] = scidb.internal.split_load_all_args(...)
%   extracts the 'version_id' key if present, returning the remaining
%   name-value pairs and the version_id as a Python-compatible value.
%   Defaults to "all".

    py_version_id = py.str('all');
    metadata_args = {};

    i = 1;
    while i <= numel(varargin)
        key = varargin{i};
        if isstring(key), key = char(key); end

        if strcmpi(key, 'version_id') && i < numel(varargin)
            val = varargin{i+1};
            if isstring(val) || ischar(val)
                py_version_id = py.str(char(val));
            elseif isnumeric(val) && isscalar(val)
                py_version_id = py.int(int64(val));
            elseif isnumeric(val) && ~isscalar(val)
                py_version_id = py.list(arrayfun(@(x) py.int(int64(x)), val, 'UniformOutput', false));
            end
            i = i + 2;
        else
            metadata_args{end+1} = varargin{i};   %#ok<AGROW>
            metadata_args{end+1} = varargin{i+1};  %#ok<AGROW>
            i = i + 2;
        end
    end
end
