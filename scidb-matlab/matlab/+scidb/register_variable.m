function register_variable(type_arg, options)
%SCIDB.REGISTER_VARIABLE  Register a variable type for database storage.
%
%   scidb.register_variable(TypeClass) registers a BaseVariable subclass.
%   The class name is extracted automatically.
%
%   scidb.register_variable(TypeClass, schema_version=2) sets a custom
%   schema version for the type.
%
%   Arguments:
%       type_arg - A BaseVariable subclass instance (e.g. RawSignal)
%
%   Name-Value Arguments:
%       schema_version - Schema version number (default 1)
%
%   Example:
%       scidb.register_variable(RawSignal);
%       scidb.register_variable(ProcessedSignal, schema_version=2);

    arguments
        type_arg
        options.schema_version (1,1) double = 1
    end

    type_name = scidb.internal.resolve_type_name(type_arg);

    py.scidb_matlab.bridge.register_matlab_variable( ...
        type_name, ...
        int64(options.schema_version));
end
