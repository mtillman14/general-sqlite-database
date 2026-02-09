function data = unwrap_input(arg)
%UNWRAP_INPUT  Extract raw MATLAB data from a thunk argument.
%
%   Used before calling feval() on the user's MATLAB function.
%   scidb.ThunkOutput and scidb.BaseVariable are unwrapped to their
%   .data property; everything else passes through unchanged.

    if isa(arg, 'scidb.ThunkOutput')
        data = arg.data;
    elseif isa(arg, 'scidb.BaseVariable')
        data = arg.data;
    else
        data = arg;
    end
end
