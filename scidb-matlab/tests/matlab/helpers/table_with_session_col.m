function [tblOut] = table_with_session_col(tblIn)
%TABLE_WITH_SESSION_COL  Double A/B and embed a non-sequential "session" column.
%   Returns tblIn with A and B doubled and a new "session" column [3;1;2].
%   The non-sequential ordering verifies that distribute= uses the column
%   values (not row indices) to determine which session each row is saved at.
tblOut = tblIn;
tblOut.A = tblOut.A * 2;
tblOut.B = tblOut.B * 2;
tblOut.session = [3; 1; 2];
end
