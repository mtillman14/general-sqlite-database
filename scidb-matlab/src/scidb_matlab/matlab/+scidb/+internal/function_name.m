function name = function_name(fcn)
%FUNCTION_NAME  Extract a clean function name from a function handle.
%
%   name = scidb.internal.function_name(@my_function)
%   Returns "my_function" as a char array.

    name = func2str(fcn);

    % Strip leading '@' if present
    if name(1) == '@'
        name = name(2:end);
    end
end
