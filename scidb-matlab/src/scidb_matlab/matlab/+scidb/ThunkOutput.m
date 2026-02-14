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
%       data         - The MATLAB result (double array, scalar, etc.)
%       py_obj       - Python thunk.core.ThunkOutput or BaseVariable (internal use)
%       record_id    - Unique record ID (populated after load, empty after compute)
%       metadata     - Metadata struct (populated after load, empty after compute)
%       content_hash - Content hash (populated after load, empty after compute)
%       lineage_hash - Lineage hash (populated after load, empty after compute)

    properties
        data            % MATLAB data (the actual computation result)
        py_obj          % Python ThunkOutput or BaseVariable (lineage shadow)
        record_id    string     % Unique record ID (populated after load)
        metadata     struct     % Metadata key-value pairs (populated after load)
        content_hash string     % Content hash (populated after load)
        lineage_hash string     % Lineage hash (populated after load)
        version_id               % Version ID (populated after load)
        parameter_id             % Parameter ID (populated after load)
    end

    methods
        function obj = ThunkOutput(matlab_data, py_thunk_output)
        %THUNKOUTPUT  Construct a ThunkOutput.
        %
        %   obj = scidb.ThunkOutput(DATA, PY_OBJ)
        %
        %   This constructor is called internally by scidb.Thunk.  Users
        %   do not create ThunkOutput objects directly.

            obj.metadata = struct();
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
                if strlength(obj.record_id) > 0
                    fprintf('  scidb.ThunkOutput [%s] containing %s\n', ...
                        obj.record_id, class(obj.data));
                else
                    fprintf('  scidb.ThunkOutput containing %s\n', ...
                        class(obj.data));
                end
                disp(obj.data);
            end
        end
    end
end
