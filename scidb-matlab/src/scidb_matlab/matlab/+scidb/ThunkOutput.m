classdef ThunkOutput < handle
%SCIDB.THUNKOUTPUT  Result of a thunked computation with lineage.
%
%   A ThunkOutput carries both the MATLAB result data and a Python
%   shadow (a real thunk.core.ThunkOutput instance) that encodes the
%   full lineage graph.
%
%   When passed to Type().save(), the Python shadow is handed directly
%   to Python's save_variable(), so isinstance checks pass and lineage
%   extraction works unchanged.
%
%   When passed as input to another scidb.Thunk call, the Python shadow
%   is placed in the inputs dict so that classify_inputs() sees a real
%   ThunkOutput and records the lineage chain.
%
%   Properties:
%       data   - The MATLAB result (double array, scalar, etc.)
%       py_obj - Python thunk.core.ThunkOutput (internal use)

    properties
        data            % MATLAB data (the actual computation result)
        py_obj          % Python ThunkOutput instance (lineage shadow)
    end

    methods
        function obj = ThunkOutput(matlab_data, py_thunk_output)
        %THUNKOUTPUT  Construct a ThunkOutput.
        %
        %   obj = scidb.ThunkOutput(DATA, PY_OBJ)
        %
        %   This constructor is called internally by scidb.Thunk.  Users
        %   do not create ThunkOutput objects directly.

            if nargin > 0
                obj.data = matlab_data;
                obj.py_obj = py_thunk_output;
            end
        end

        function disp(obj)
        %DISP  Display the ThunkOutput.

            if isempty(obj.data)
                fprintf('  scidb.ThunkOutput (empty)\n');
            else
                fprintf('  scidb.ThunkOutput containing %s\n', ...
                    class(obj.data));
                disp(obj.data);
            end
        end
    end
end
