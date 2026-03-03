function result_tbl = for_each(fn, inputs, outputs, varargin)
%SCIHIST.FOR_EACH  Lineage-tracked for_each — auto-wraps fn in Thunk.
%
%   scihist.for_each(@FN, INPUTS, OUTPUTS, Name, Value, ...)
%
%   This is Layer 3 (lineage tracking). It:
%   1. Auto-wraps the function handle in scidb.Thunk for lineage tracking
%   2. Wraps the Thunk in a plain function handle for scidb.for_each
%   3. Delegates to scidb.for_each for all DB I/O and loop orchestration
%
%   Arguments:
%       fn      - Function handle (auto-wrapped in Thunk if not already one)
%       inputs  - Struct mapping parameter names to BaseVariable instances,
%                 scidb.Fixed wrappers, scidb.Merge wrappers,
%                 scifor.PathInput instances, or constant values.
%       outputs - Cell array of BaseVariable instances for output types
%
%   Name-Value Arguments: same as scidb.for_each
%
%   Returns:
%       result_tbl - MATLAB table with metadata columns and output columns.
%
%   Example:
%       scihist.for_each(@filter_data, ...
%           struct('step_length', StepLength(), 'smoothing', 0.2), ...
%           {FilteredStepLength()}, ...
%           subject=[1 2 3], session=["A" "B"]);

    % Auto-wrap in Thunk if not already one
    if isa(fn, 'scidb.Thunk')
        thunk_obj = fn;
    else
        thunk_obj = scidb.Thunk(fn);
    end

    % Wrap Thunk in a plain function handle for scidb.for_each
    fn_plain = @(varargin) thunk_obj(varargin{:});

    % Delegate to scidb.for_each
    result_tbl = scidb.for_each(fn_plain, inputs, outputs, varargin{:});
end
