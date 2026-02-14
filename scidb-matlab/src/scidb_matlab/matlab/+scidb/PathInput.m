classdef PathInput < handle
%SCIDB.PATHINPUT  Resolve a path template using iteration metadata.
%
%   Works as an input to scidb.for_each: on each iteration, .load()
%   substitutes the current metadata values into the template and
%   returns the resolved file path as a string.  The user's function
%   receives the path and handles file reading itself.
%
%   PI = scidb.PathInput(TEMPLATE)
%   PI = scidb.PathInput(TEMPLATE, root_folder=FOLDER)
%
%   The template uses {key} placeholders that are replaced by the
%   metadata values supplied by for_each on each iteration.
%
%   Example:
%       scidb.for_each(@process_file, ...
%           struct('filepath', scidb.PathInput("{subject}/trial_{trial}.mat", ...
%                                              root_folder="/data"), ...
%                  'baseline', StepLength()), ...
%           {ProcessedSignal()}, ...
%           subject=[1 2 3], ...
%           trial=[0 1 2]);

    properties (SetAccess = private)
        path_template  string   % Format string with {key} placeholders
        root_folder    string   % Optional root directory
    end

    methods
        function obj = PathInput(path_template, options)
        %PATHINPUT  Construct a PathInput.
        %
        %   PI = scidb.PathInput(TEMPLATE)
        %   PI = scidb.PathInput(TEMPLATE, root_folder=FOLDER)

            arguments
                path_template  string
                options.root_folder  string = ""
            end

            obj.path_template = path_template;
            obj.root_folder = options.root_folder;
        end

        function filepath = load(obj, varargin)
        %LOAD  Resolve the template with the given metadata.
        %
        %   PATH = pi.load(Name, Value, ...)
        %
        %   Substitutes {key} placeholders in the template with the
        %   supplied metadata values and returns the resolved absolute
        %   path as a string.  The 'db' key is accepted and ignored
        %   for compatibility with for_each's uniform db= passthrough.

            % Parse name-value pairs
            if mod(numel(varargin), 2) ~= 0
                error('scidb:PathInput', ...
                    'Metadata arguments must be name-value pairs.');
            end

            resolved = obj.path_template;
            for i = 1:2:numel(varargin)
                key = string(varargin{i});
                if strcmpi(key, 'db')
                    continue;  % Skip db parameter
                end
                val = varargin{i+1};
                if isnumeric(val)
                    val_str = num2str(val);
                else
                    val_str = string(val);
                end
                resolved = strrep(resolved, "{" + key + "}", val_str);
            end

            % Resolve to absolute path
            if strlength(obj.root_folder) > 0
                filepath = string(fullfile(obj.root_folder, resolved));
            else
                filepath = string(fullfile(pwd, resolved));
            end
        end

        function disp(obj)
        %DISP  Display the PathInput.
            if strlength(obj.root_folder) > 0
                fprintf('  scidb.PathInput("%s", root_folder="%s")\n', ...
                    obj.path_template, obj.root_folder);
            else
                fprintf('  scidb.PathInput("%s")\n', obj.path_template);
            end
        end
    end
end
