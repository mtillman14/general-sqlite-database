function n = count_elements(x)
%COUNT_ELEMENTS  Test function: return the number of elements as a double.
%   Works for any type (arrays, string arrays, cell arrays, etc.).
    n = double(numel(x));
end
