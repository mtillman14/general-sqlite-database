function y = sum_table_cols(data)
%SUM_TABLE_COLS  Test function: sum all numeric columns of a table row-wise.
    nums = varfun(@double, data);
    y = sum(nums{:,:}, 2);
end
