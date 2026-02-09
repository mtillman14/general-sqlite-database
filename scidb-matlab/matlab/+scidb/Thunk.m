classdef Thunk < handle
%SCIDB.THUNK  Wrap a MATLAB function for lineage tracking and caching.
%
%   T = scidb.Thunk(@my_function) wraps a named MATLAB function.
%   When called, T checks the database for a cached result.  On cache
%   miss it executes the function and returns a scidb.ThunkOutput that
%   carries both the MATLAB result and a Python lineage shadow.
%
%   T = scidb.Thunk(@my_function, unpack_output=true) enables multi-output
%   mode where the function must return a cell array and each element
%   becomes a separate ThunkOutput.
%
%   Usage:
%       filter_fn = scidb.Thunk(@bandpass_filter);
%       result = filter_fn(raw_signal, 10, 200);
%       scidb.save("FilteredSignal", result, subject=1, session="A");
%
%   Multi-output:
%       split_fn = scidb.Thunk(@split_data, unpack_output=true);
%       [first, second] = split_fn(data);
%
%   Notes:
%     - Only named functions are supported (no anonymous functions).
%     - The function is identified by a SHA-256 hash of its .m source file.
%     - All lineage computation, cache checking, and hashing is delegated
%       to Python via the py. interface.

    properties (SetAccess = private)
        fcn           function_handle   % The wrapped MATLAB function
        unpack_output logical           % Multi-output mode
        py_thunk                        % Python MatlabThunk proxy
    end

    methods
        function obj = Thunk(fcn, options)
        %THUNK  Construct a thunk wrapper around a MATLAB function.

            arguments
                fcn           function_handle
                options.unpack_output logical = false
            end

            obj.fcn = fcn;
            obj.unpack_output = options.unpack_output;

            % Compute function identity hash from source
            source_hash = scidb.internal.hash_function(fcn);
            func_name = scidb.internal.function_name(fcn);

            % Create the Python-side proxy
            obj.py_thunk = py.scidb_matlab.bridge.MatlabThunk( ...
                source_hash, func_name, options.unpack_output);
        end

        function varargout = subsref(obj, s)
        %SUBSREF  Overload () to make T(args...) work as a thunk call.

            if strcmp(s(1).type, '()')
                args = s(1).subs;
                [varargout{1:nargout}] = invoke(obj, args);
            else
                % Delegate dot-access and {} to default behaviour
                [varargout{1:nargout}] = builtin('subsref', obj, s);
            end
        end
    end

    methods (Access = private)
        function varargout = invoke(obj, args)
        %INVOKE  Core thunk call: check cache, execute on miss, wrap output.

            % --- Step 1: Build Python inputs dict ---
            py_inputs = py.dict();
            for i = 1:numel(args)
                key = sprintf('arg_%d', i - 1);
                py_inputs{key} = scidb.internal.to_python_input(args{i});
            end

            % --- Step 2: Create pipeline thunk & check cache ---
            py_pt = py.scidb_matlab.bridge.MatlabPipelineThunk( ...
                obj.py_thunk, py_inputs);

            cached = py.scidb_matlab.bridge.check_cache(py_pt);

            if ~isa(cached, 'py.NoneType') && ~isempty(cached)
                % --- Cache HIT ---
                varargout = wrap_cached(obj, py_pt, cached);
            else
                % --- Cache MISS: execute in MATLAB ---
                varargout = execute_and_wrap(obj, py_pt, args);
            end
        end

        function out = wrap_cached(obj, py_pt, cached)
        %WRAP_CACHED  Convert cached Python values to MATLAB ThunkOutputs.

            cached_cell = cell(cached);
            n = numel(cached_cell);

            if obj.unpack_output
                out = cell(1, n);
                for i = 1:n
                    py_to = py.scidb_matlab.bridge.make_thunk_output( ...
                        py_pt, int64(i - 1), cached_cell{i});
                    out{i} = scidb.ThunkOutput( ...
                        scidb.internal.from_python(cached_cell{i}), py_to);
                end
            else
                py_to = py.scidb_matlab.bridge.make_thunk_output( ...
                    py_pt, int64(0), cached_cell{1});
                out = {scidb.ThunkOutput( ...
                    scidb.internal.from_python(cached_cell{1}), py_to)};
            end
        end

        function out = execute_and_wrap(obj, py_pt, args)
        %EXECUTE_AND_WRAP  Run the MATLAB function and wrap results.

            % Unwrap inputs to raw MATLAB data for execution
            matlab_args = cell(1, numel(args));
            for i = 1:numel(args)
                matlab_args{i} = scidb.internal.unwrap_input(args{i});
            end

            % Execute
            if obj.unpack_output
                % Multi-output: function must return a cell array
                result = feval(obj.fcn, matlab_args{:});
                if ~iscell(result)
                    error('scidb:UnpackError', ...
                        ['Function %s has unpack_output=true but did not ' ...
                         'return a cell array.'], func2str(obj.fcn));
                end
                n = numel(result);
                out = cell(1, n);
                for i = 1:n
                    py_data = scidb.internal.to_python(result{i});
                    py_to = py.scidb_matlab.bridge.make_thunk_output( ...
                        py_pt, int64(i - 1), py_data);
                    out{i} = scidb.ThunkOutput(result{i}, py_to);
                end
            else
                result = feval(obj.fcn, matlab_args{:});
                py_data = scidb.internal.to_python(result);
                py_to = py.scidb_matlab.bridge.make_thunk_output( ...
                    py_pt, int64(0), py_data);
                out = {scidb.ThunkOutput(result, py_to)};
            end
        end
    end
end
