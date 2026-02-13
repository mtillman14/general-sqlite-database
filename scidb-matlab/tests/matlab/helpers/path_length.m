function y = path_length(filepath)
%PATH_LENGTH  Test function: return the length of a file path as a double.
    y = double(strlength(string(filepath)));
end
