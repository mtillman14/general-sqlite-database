function col = normalize_cell_column(col_data)
%NORMALIZE_CELL_COLUMN  Convert a cell column to its native type.
%   Attempts to collapse a cell array into a more specific type:
%   - All scalar numeric/logical -> numeric vector via cell2mat
%   - All scalar strings/chars -> string array
%   - All scalar structs with identical fields -> struct array
%   - Otherwise returns the cell array unchanged
    n = numel(col_data);
    all_scalar_numeric = true;
    all_string = true;
    all_scalar_struct = n > 0;
    ref_fields = {};
    for i = 1:n
        v = col_data{i};
        if ~((isnumeric(v) || islogical(v)) && isscalar(v))
            all_scalar_numeric = false;
        end
        if ~((isstring(v) && isscalar(v)) || ischar(v))
            all_string = false;
        end
        if all_scalar_struct
            if isstruct(v) && isscalar(v)
                if i == 1
                    ref_fields = sort(fieldnames(v));
                elseif ~isequal(sort(fieldnames(v)), ref_fields)
                    all_scalar_struct = false;
                end
            else
                all_scalar_struct = false;
            end
        end
    end
    if all_scalar_numeric
        col = cell2mat(col_data);
    elseif all_string
        col = string(col_data);
    elseif all_scalar_struct
        col = reshape([col_data{:}], [], 1);
    else
        col = col_data;
    end
end
