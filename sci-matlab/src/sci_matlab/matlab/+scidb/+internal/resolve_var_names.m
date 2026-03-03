function py_names = resolve_var_names(variables)
%RESOLVE_VAR_NAMES  Convert MATLAB variable references to a Python list of name strings.
%
%   Accepts:
%       - A single BaseVariable instance   e.g. StepLength()
%       - A single char vector             e.g. 'StepLength'
%       - A single string scalar           e.g. "StepLength"
%       - A cell array of any of the above e.g. {StepLength(), 'StepWidth'}
%       - A string array                   e.g. ["StepLength", "StepWidth"]

    if isstring(variables) && ~isscalar(variables)
        % String array -> cell array of chars
        names = cellstr(variables);
    elseif isstring(variables) && isscalar(variables)
        names = {char(variables)};
    elseif ischar(variables)
        names = {variables};
    elseif iscell(variables)
        names = cell(1, numel(variables));
        for i = 1:numel(variables)
            v = variables{i};
            if ischar(v)
                names{i} = v;
            elseif isstring(v)
                names{i} = char(v);
            elseif isa(v, 'scidb.BaseVariable')
                names{i} = class(v);
            else
                error('scidb:InvalidInput', ...
                    'Expected a string or BaseVariable, got "%s".', class(v));
            end
        end
    elseif isa(variables, 'scidb.BaseVariable')
        names = {class(variables)};
    else
        error('scidb:InvalidInput', ...
            'Expected a string, cell array, or BaseVariable, got "%s".', class(variables));
    end

    py_names = py.list(names);
end
