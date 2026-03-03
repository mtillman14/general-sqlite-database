function data = unwrap_input(arg)
%UNWRAP_INPUT  Extract raw MATLAB data from a thunk argument.
%
%   Used before calling feval() on the user's MATLAB function.
%   scidb.ThunkOutput and scidb.BaseVariable are unwrapped to their
%   .data property; everything else passes through unchanged.
%
%   When the argument is an array of ThunkOutput or BaseVariable (e.g.
%   multiple matches from load()), a cell array of all .data values is
%   returned so that the calling function receives all results.

    if isa(arg, 'scidb.ThunkOutput') || isa(arg, 'scidb.BaseVariable')
        if numel(arg) > 1
            data = cell(1, numel(arg));
            all_numeric_scalar = true;
            for i = 1:numel(arg)
                data{i} = arg(i).data;
                if ~(isnumeric(data{i}) && isscalar(data{i}))
                    all_numeric_scalar = false;
                end
            end
            if all_numeric_scalar
                data = cell2mat(data);
            end
        else
            data = arg.data;
        end
    else
        data = arg;
    end
end
