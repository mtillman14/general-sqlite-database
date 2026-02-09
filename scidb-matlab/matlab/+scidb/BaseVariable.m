classdef BaseVariable < handle
%SCIDB.BASEVARIABLE  Container for data loaded from the database.
%
%   Returned by scidb.load() and scidb.load_all().  Carries the loaded
%   MATLAB data together with metadata and a Python shadow object that
%   allows seamless participation in the thunk lineage system.
%
%   Properties (user-facing):
%       data         - The loaded data (MATLAB native type)
%       record_id    - Unique record identifier (string)
%       metadata     - Struct of metadata key-value pairs
%       content_hash - Content hash of the data (string)
%       lineage_hash - Lineage hash, if data was computed by a thunk (string)
%
%   Properties (internal):
%       py_obj       - Python BaseVariable instance (used by Thunk and save)
%
%   Users do not construct BaseVariable directly; use scidb.load instead.

    properties
        data                    % MATLAB data
        record_id    string     % Unique record ID
        metadata     struct     % Metadata key-value pairs
        content_hash string     % Content hash (16-char hex)
        lineage_hash string     % Lineage hash (64-char hex), empty if raw data
        py_obj                  % Python BaseVariable shadow (internal)
    end

    methods
        function obj = BaseVariable()
        %BASEVARIABLE  Construct an empty BaseVariable.
            obj.metadata = struct();
        end

        function disp(obj)
        %DISP  Display the BaseVariable.
            if isempty(obj.data)
                fprintf('  scidb.BaseVariable (empty)\n');
            else
                fprintf('  scidb.BaseVariable [%s]\n', obj.record_id);
                fprintf('    data: %s\n', class(obj.data));
                if ~isempty(fieldnames(obj.metadata))
                    fprintf('    metadata: ');
                    disp(obj.metadata);
                end
            end
        end
    end

    methods (Static)
        function objs = empty()
        %EMPTY  Create an empty BaseVariable array (for preallocation).
            objs = scidb.BaseVariable.empty(0, 0);
        end
    end
end
