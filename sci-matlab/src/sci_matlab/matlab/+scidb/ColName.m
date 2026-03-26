classdef ColName
%SCIDB.COLNAME  Resolves to the single data column name of a Variable type.
%
%   Marker that resolves to the single data column name of a DB-backed
%   variable at for_each time. The function body stays framework-agnostic.
%
%   Properties:
%       var_type - BaseVariable instance whose data column name will be resolved
%
%   Example:
%       scidb.for_each(@analyze, ...
%           struct('table', MyVar(), 'col_name', scidb.ColName(MyVar())), ...
%           {Result()}, ...
%           subject=[1 2 3])
%
%       % The function is pure:
%       function out = analyze(table, col_name)
%           out = mean(table.(col_name));
%       end

    properties (SetAccess = private)
        var_type  % BaseVariable instance
    end

    methods
        function obj = ColName(var_type)
        %COLNAME  Construct a ColName wrapper for DB-backed variables.
        %
        %   C = scidb.ColName(VarInstance())
        %
        %   Arguments:
        %       var_type - A BaseVariable instance

            obj.var_type = var_type;
        end

        function disp(obj)
        %DISP  Display the ColName wrapper.
            fprintf('  scidb.ColName(%s)\n', class(obj.var_type));
        end
    end
end
