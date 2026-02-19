function [tableIn] = double_table_values(tableIn)
%DOUBLE_TABLE_VALUES: Double the values in column A and B in a table
tableIn.A = tableIn.A * 2;
tableIn.B = tableIn.B * 2;
end