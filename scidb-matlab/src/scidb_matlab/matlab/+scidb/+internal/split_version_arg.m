function [metadata_args, version, db] = split_version_arg(varargin)
%SPLIT_VERSION_ARG  Separate 'version' and 'db' from other name-value metadata.
%
%   [metadata_args, version, db] = scidb.internal.split_version_arg(...)
%   extracts the 'version' and 'db' keys if present, returning the
%   remaining name-value pairs, the version string, and the db value.
%   Defaults to "latest" and [] respectively.

    version = "latest";
    db = [];
    metadata_args = {};

    i = 1;
    while i <= numel(varargin)
        key = varargin{i};
        if isstring(key), key = char(key); end

        if strcmpi(key, 'version') && i < numel(varargin)
            version = string(varargin{i+1});
            i = i + 2;
        elseif strcmpi(key, 'db') && i < numel(varargin)
            db = varargin{i+1};
            i = i + 2;
        else
            metadata_args{end+1} = varargin{i};   %#ok<AGROW>
            metadata_args{end+1} = varargin{i+1};  %#ok<AGROW>
            i = i + 2;
        end
    end
end
