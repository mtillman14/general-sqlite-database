function result = split_data(x)
%SPLIT_DATA  Test function for unpack_output: split array into two halves.
%   Returns a cell array {first_half, second_half}.
    n = floor(numel(x) / 2);
    result = {x(1:n), x(n+1:end)};
end
