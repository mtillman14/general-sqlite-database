classdef PathGenerator < handle
%SCIDB.PATHGENERATOR  Generate file paths from a template and metadata.
%
%   Generate all combinations of metadata values and resolve them into
%   fully-qualified file paths using a template string.
%
%   PG = scidb.PathGenerator(TEMPLATE, Name, Value, ...)
%   PG = scidb.PathGenerator(TEMPLATE, root_folder=FOLDER, Name, Value, ...)
%
%   The template uses {key} placeholders that are replaced by each
%   metadata combination.
%
%   Properties:
%       paths    - String array of resolved file paths
%       metadata - Struct array of metadata for each path
%       n        - Number of path/metadata combinations
%
%   Methods:
%       length   - Number of combinations (same as .n)
%       get      - Get the i-th path and metadata: [p, m] = pg.get(i)
%       to_table - Return paths and metadata as a MATLAB table
%
%   Example:
%       pg = scidb.PathGenerator("{subject}/trial_{trial}.mat", ...
%           root_folder="/data/experiment", ...
%           subject=0:2, ...
%           trial=0:4);
%
%       % Iterate over all combinations
%       for i = 1:length(pg)
%           [filepath, meta] = pg.get(i);
%           fprintf('%s  subject=%d trial=%d\n', filepath, ...
%               meta.subject, meta.trial);
%       end
%
%       % Access properties directly
%       disp(pg.paths);
%       disp(pg.metadata);

    properties (SetAccess = private)
        paths    string     % Resolved file paths (string array)
        metadata struct     % Metadata structs (struct array)
        n        double     % Number of combinations
    end

    properties (Access = private)
        path_template  string
        root_folder    string
        metadata_keys  string
    end

    methods
        function obj = PathGenerator(path_template, varargin)
        %PATHGENERATOR  Construct a PathGenerator.
        %
        %   PG = scidb.PathGenerator(TEMPLATE, Name, Value, ...)
        %
        %   Arguments:
        %       path_template - Format string with {key} placeholders
        %
        %   Name-Value Arguments:
        %       root_folder   - Optional root folder (paths resolved
        %                       relative to this; default: pwd)
        %       (any other)   - Metadata iterables (arrays of values)

            obj.path_template = string(path_template);

            % Separate root_folder from metadata name-value pairs
            [meta_args, root] = scidb.PathGenerator.split_root_arg(varargin{:});
            obj.root_folder = root;

            % Parse metadata name-value pairs
            if mod(numel(meta_args), 2) ~= 0
                error('scidb:PathGenerator', ...
                    'Metadata arguments must be name-value pairs.');
            end

            keys = string.empty;
            values = {};
            for i = 1:2:numel(meta_args)
                keys(end+1) = string(meta_args{i}); %#ok<AGROW>
                v = meta_args{i+1};
                % Normalize to cell array for uniform handling
                if isnumeric(v)
                    values{end+1} = num2cell(v); %#ok<AGROW>
                elseif isstring(v)
                    values{end+1} = cellstr(v); %#ok<AGROW>
                elseif iscell(v)
                    values{end+1} = v; %#ok<AGROW>
                else
                    values{end+1} = {v}; %#ok<AGROW>
                end
            end

            obj.metadata_keys = keys;

            % Compute Cartesian product of all metadata values
            if isempty(values)
                combos = {{}};
            else
                combos = scidb.PathGenerator.cartesian_product(values);
            end

            obj.n = numel(combos);
            obj.paths = strings(1, obj.n);
            obj.metadata = repmat(struct(), 1, obj.n);

            for idx = 1:obj.n
                combo = combos{idx};

                % Build metadata struct for this combination
                meta = struct();
                resolved = obj.path_template;
                for k = 1:numel(keys)
                    val = combo{k};
                    meta.(keys(k)) = val;

                    % Replace {key} placeholder with value
                    if isnumeric(val)
                        val_str = num2str(val);
                    else
                        val_str = string(val);
                    end
                    resolved = strrep(resolved, "{" + keys(k) + "}", val_str);
                end

                % Resolve to absolute path
                if strlength(obj.root_folder) > 0
                    full_path = fullfile(obj.root_folder, resolved);
                else
                    full_path = fullfile(pwd, resolved);
                end

                obj.paths(idx) = string(full_path);
                obj.metadata(idx) = meta;
            end
        end

        function n = length(obj)
        %LENGTH  Number of path/metadata combinations.
            n = obj.n;
        end

        function [path, meta] = get(obj, index)
        %GET  Get the i-th path and metadata.
        %
        %   [PATH, META] = pg.get(I)
        %
        %   Returns the path as a string and the metadata as a struct.
            if index < 1 || index > obj.n
                error('scidb:PathGenerator', ...
                    'Index %d out of range [1, %d].', index, obj.n);
            end
            path = obj.paths(index);
            meta = obj.metadata(index);
        end

        function t = to_table(obj)
        %TO_TABLE  Return all combinations as a MATLAB table.
        %
        %   T = pg.to_table()
        %
        %   The table has a 'path' column plus one column per metadata key.
            t = table(obj.paths(:), 'VariableNames', {'path'});
            for k = 1:numel(obj.metadata_keys)
                key = obj.metadata_keys(k);
                vals = {obj.metadata.(key)}';
                % Convert to native column if possible
                if all(cellfun(@isnumeric, vals))
                    t.(key) = cell2mat(vals);
                else
                    t.(key) = string(vals);
                end
            end
        end

        function disp(obj)
        %DISP  Display the PathGenerator.
            fprintf('  scidb.PathGenerator (%d paths)\n', obj.n);
            fprintf('    template: %s\n', obj.path_template);
            if strlength(obj.root_folder) > 0
                fprintf('    root:     %s\n', obj.root_folder);
            end
            if ~isempty(obj.metadata_keys)
                fprintf('    keys:     %s\n', strjoin(obj.metadata_keys, ', '));
            end
            show_n = min(obj.n, 5);
            for i = 1:show_n
                fprintf('    [%d] %s\n', i, obj.paths(i));
            end
            if obj.n > show_n
                fprintf('    ... and %d more\n', obj.n - show_n);
            end
        end
    end

    methods (Static, Access = private)
        function [meta_args, root] = split_root_arg(varargin)
        %SPLIT_ROOT_ARG  Extract root_folder from name-value pairs.
            root = "";
            meta_args = {};
            i = 1;
            while i <= numel(varargin)
                if ischar(varargin{i}) || isstring(varargin{i})
                    if strcmpi(varargin{i}, 'root_folder')
                        root = string(varargin{i+1});
                        i = i + 2;
                        continue;
                    end
                end
                meta_args{end+1} = varargin{i}; %#ok<AGROW>
                i = i + 1;
            end
        end

        function combos = cartesian_product(value_cells)
        %CARTESIAN_PRODUCT  Compute Cartesian product of cell arrays.
        %
        %   Each element of value_cells is a cell array of values.
        %   Returns a cell array of cell arrays, one per combination.
            n = numel(value_cells);
            if n == 0
                combos = {{}};
                return;
            end

            % Build index grids using ndgrid
            sizes = cellfun(@numel, value_cells);
            idx_args = arrayfun(@(s) 1:s, sizes, 'UniformOutput', false);
            grids = cell(1, n);
            [grids{:}] = ndgrid(idx_args{:});

            total = prod(sizes);
            combos = cell(1, total);
            for t = 1:total
                combo = cell(1, n);
                for d = 1:n
                    combo{d} = value_cells{d}{grids{d}(t)};
                end
                combos{t} = combo;
            end
        end
    end
end
