classdef Merge
%SCIDB.MERGE  Combine multiple variables into a single table input for for_each.
%
%   Merges 2+ variable inputs into a single MATLAB table that is passed
%   as one argument to the function. Each constituent is loaded individually
%   per iteration, converted to a keyed table (schema key columns + data
%   columns), and inner-joined on common schema keys.
%
%   When constituents return multiple records (e.g. iterating at a coarse
%   level), the records are joined by their shared schema keys. Unmatched
%   rows are dropped (inner join). When each constituent returns exactly
%   one record, this reduces to a simple column-wise merge.
%
%   Table variables contribute all their columns. Array and scalar variables
%   are added as a column named after the variable class name.
%
%   Constituents can be:
%   - BaseVariable instances (loaded from the database)
%   - scidb.Fixed wrappers (loaded with overridden metadata)
%   - BaseVariable instances with column selection (e.g. MyVar("col"))
%   - Combinations: scidb.Fixed(MyVar("col"), session="BL")
%
%   Properties:
%       var_specs - Cell array of variable specs to merge
%
%   Example:
%       % Merge a table with a computed column
%       scidb.for_each(@analyze, ...
%           struct('data', scidb.Merge(GaitData(), PareticSide())), ...
%           {Result()}, ...
%           subject=[1 2 3]);
%
%       % Merge with Fixed override
%       scidb.for_each(@analyze, ...
%           struct('data', scidb.Merge( ...
%               GaitData("force"), ...
%               scidb.Fixed(PareticSide(), session="BL"))), ...
%           {Result()}, ...
%           subject=[1 2 3], session=["A" "B"]);
%
%       % Multi-record merge (joined by schema keys)
%       scidb.for_each(@analyze, ...
%           struct('data', scidb.Merge(GaitData(), PareticSide())), ...
%           {Result()}, ...
%           subject=[1 2 3]);

    properties (SetAccess = private)
        var_specs  cell  % Cell array of variable specs
    end

    methods
        function obj = Merge(varargin)
        %MERGE  Construct a Merge wrapper.
        %
        %   M = scidb.Merge(VarA(), VarB(), ...)
        %
        %   Arguments:
        %       2+ variable specs: BaseVariable instances, Fixed wrappers,
        %       or BaseVariable instances with column selection.

            if nargin < 2
                error('scidb:Merge', ...
                    'Merge requires at least 2 variable inputs, got %d.', nargin);
            end

            for i = 1:nargin
                if isa(varargin{i}, 'scidb.Merge')
                    error('scidb:Merge', ...
                        'Cannot nest Merge inside another Merge.');
                end
            end

            obj.var_specs = varargin;
        end

        function disp(obj)
        %DISP  Display the Merge wrapper.
            parts = cell(1, numel(obj.var_specs));
            for i = 1:numel(obj.var_specs)
                spec = obj.var_specs{i};
                if isa(spec, 'scidb.Fixed')
                    inner = spec.var_type;
                    inner_name = class(inner);
                    fields = fieldnames(spec.fixed_metadata);
                    fp = cell(1, numel(fields));
                    for f = 1:numel(fields)
                        val = spec.fixed_metadata.(fields{f});
                        if isnumeric(val)
                            fp{f} = sprintf('%s=%g', fields{f}, val);
                        else
                            fp{f} = sprintf('%s="%s"', fields{f}, string(val));
                        end
                    end
                    parts{i} = sprintf('Fixed(%s, %s)', inner_name, strjoin(fp, ', '));
                else
                    parts{i} = class(spec);
                end
            end
            fprintf('  scidb.Merge(%s)\n', strjoin(parts, ', '));
        end
    end
end
