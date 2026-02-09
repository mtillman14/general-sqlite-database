function type_name = resolve_type_name(type_arg)
%RESOLVE_TYPE_NAME  Extract the variable type name from a class token.
%
%   type_name = scidb.internal.resolve_type_name(RawSignal)
%
%   Accepts a BaseVariable subclass instance (class token) and returns
%   the class name as a char.  This enables the calling pattern:
%
%       scidb.save(RawSignal, data, subject=1)
%
%   where MATLAB implicitly constructs a RawSignal() instance.

    if isa(type_arg, 'scidb.BaseVariable')
        type_name = class(type_arg);
    else
        error('scidb:InvalidType', ...
            ['First argument must be a BaseVariable subclass (e.g. RawSignal).\n' ...
             'Got: %s'], class(type_arg));
    end
end
