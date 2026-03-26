classdef ColName
%SCIFOR.COLNAME  Resolves to the single non-schema data column name of a table.
%
%   Marker that resolves to the single non-schema data column name
%   at for_each time. The function body stays framework-agnostic.
%
%   Properties:
%       data - MATLAB table whose single data column name will be resolved
%
%   Example:
%       result = scifor.for_each(@analyze, ...
%           struct('table', data_table, 'col_name', scifor.ColName(data_table)), ...
%           subject=[1 2 3])
%
%       % The function is pure:
%       function out = analyze(table, col_name)
%           out = mean(table.(col_name));
%       end

    properties (SetAccess = private)
        data  % MATLAB table
    end

    methods
        function obj = ColName(data)
        %COLNAME  Construct a ColName wrapper.
        %
        %   C = scifor.ColName(tbl)
        %
        %   Arguments:
        %       data - A MATLAB table with exactly one non-schema data column

            obj.data = data;
        end

        function disp(obj)
        %DISP  Display the ColName wrapper.
            if istable(obj.data)
                fprintf('  scifor.ColName(<table %dx%d>)\n', ...
                    height(obj.data), width(obj.data));
            else
                fprintf('  scifor.ColName(%s)\n', class(obj.data));
            end
        end
    end
end
