classdef Fixed
%SCIDB.FIXED  Specify fixed metadata overrides for a for_each input.
%
%   Use this when an input should be loaded with different metadata
%   than the current iteration's metadata.
%
%   Properties:
%       var_type       - Variable instance (e.g. StepLength())
%       fixed_metadata - Struct of metadata values to override
%
%   Example:
%       % Always load baseline from session="BL", regardless of current session
%       scidb.for_each(@compare_to_baseline, ...
%           struct('baseline', scidb.Fixed(StepLength(), session="BL"), ...
%                  'current',  StepLength()), ...
%           {Delta()}, ...
%           subject=[1 2 3], ...
%           session=["A" "B"]);

    properties (SetAccess = private)
        var_type        % BaseVariable instance (for type resolution)
        fixed_metadata  struct  % Metadata overrides
    end

    methods
        function obj = Fixed(var_type, varargin)
        %FIXED  Construct a Fixed metadata wrapper.
        %
        %   F = scidb.Fixed(TypeInstance(), Name, Value, ...)
        %
        %   Arguments:
        %       var_type - A BaseVariable instance (e.g. StepLength())
        %
        %   Name-Value Arguments:
        %       Metadata keys and their fixed values

            obj.var_type = var_type;

            if mod(numel(varargin), 2) ~= 0
                error('scidb:Fixed', ...
                    'Fixed metadata must be name-value pairs.');
            end

            s = struct();
            for i = 1:2:numel(varargin)
                s.(string(varargin{i})) = varargin{i+1};
            end
            obj.fixed_metadata = s;
        end

        function disp(obj)
        %DISP  Display the Fixed wrapper.
            type_name = class(obj.var_type);
            fields = fieldnames(obj.fixed_metadata);
            if isempty(fields)
                fprintf('  scidb.Fixed(%s)\n', type_name);
            else
                parts = cell(1, numel(fields));
                for i = 1:numel(fields)
                    val = obj.fixed_metadata.(fields{i});
                    if isnumeric(val)
                        parts{i} = sprintf('%s=%g', fields{i}, val);
                    else
                        parts{i} = sprintf('%s="%s"', fields{i}, string(val));
                    end
                end
                fprintf('  scidb.Fixed(%s, %s)\n', type_name, strjoin(parts, ', '));
            end
        end
    end
end
