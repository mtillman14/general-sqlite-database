function for_each(fn, inputs, outputs, varargin)
%SCIDB.FOR_EACH  Execute a function for all combinations of metadata.
%
%   scidb.for_each(@FN, INPUTS, OUTPUTS, Name, Value, ...)
%
%   Iterates over every combination of the supplied metadata values.
%   For each combination it loads the specified input variables, calls
%   the function, and saves the results under the corresponding output
%   variable types.
%
%   Arguments:
%       fn      - Function handle or scidb.Thunk
%       inputs  - Struct mapping parameter names to BaseVariable instances
%                 (or scidb.Fixed wrappers for overridden metadata).
%                 The field order determines argument order when calling fn.
%       outputs - Cell array of BaseVariable instances for output types
%
%   Name-Value Arguments:
%       dry_run       - If true, preview without executing (default: false)
%       save          - If true, save outputs (default: true)
%       pass_metadata - If true, pass metadata as trailing name-value
%                       arguments to fn. If not specified, auto-detects
%                       based on fn.generates_file when fn is a Thunk.
%       (any other)   - Metadata iterables (numeric or string arrays)
%
%   Example:
%       scidb.for_each(@filter_data, ...
%           struct('step_length', StepLength(), ...
%                  'step_width',  StepWidth()), ...
%           {FilteredStepLength(), FilteredStepWidth()}, ...
%           subject=[1 2 3], ...
%           session=["A" "B"]);
%
%       % Preview what would happen
%       scidb.for_each(@filter_data, ...
%           struct('step_length', StepLength()), ...
%           {FilteredStepLength()}, ...
%           dry_run=true, ...
%           subject=[1 2 3]);
%
%       % With Fixed inputs (always load baseline from session "BL")
%       scidb.for_each(@compare_to_baseline, ...
%           struct('baseline', scidb.Fixed(StepLength(), session="BL"), ...
%                  'current',  StepLength()), ...
%           {Delta()}, ...
%           subject=[1 2 3], ...
%           session=["A" "B"]);

    % --- Parse options vs metadata name-value pairs ---
    [meta_args, opts] = split_options(varargin{:});

    dry_run = opts.dry_run;
    do_save = opts.save;

    % Auto-detect pass_metadata
    if isempty(opts.pass_metadata)
        if isa(fn, 'scidb.Thunk')
            should_pass_metadata = false;  % Thunks don't take metadata
        else
            should_pass_metadata = false;  % Default off for regular functions
        end
    else
        should_pass_metadata = opts.pass_metadata;
    end

    % Get function name for display
    if isa(fn, 'scidb.Thunk')
        fn_name = func2str(fn.fcn);
    elseif isa(fn, 'function_handle')
        fn_name = func2str(fn);
    else
        fn_name = 'unknown';
    end

    % Parse metadata iterables
    if mod(numel(meta_args), 2) ~= 0
        error('scidb:for_each', 'Metadata arguments must be name-value pairs.');
    end

    meta_keys = string.empty;
    meta_values = {};
    for i = 1:2:numel(meta_args)
        meta_keys(end+1) = string(meta_args{i}); %#ok<AGROW>
        v = meta_args{i+1};
        if isnumeric(v)
            meta_values{end+1} = num2cell(v); %#ok<AGROW>
        elseif isstring(v)
            meta_values{end+1} = cellstr(v); %#ok<AGROW>
        elseif iscell(v)
            meta_values{end+1} = v; %#ok<AGROW>
        else
            meta_values{end+1} = {v}; %#ok<AGROW>
        end
    end

    % Compute total iterations
    total = 1;
    for i = 1:numel(meta_values)
        total = total * numel(meta_values{i});
    end

    % Parse inputs struct
    input_names = fieldnames(inputs);
    n_inputs = numel(input_names);

    % Parse outputs cell array
    n_outputs = numel(outputs);

    % --- Dry-run header ---
    if dry_run
        fprintf('[dry-run] for_each(%s)\n', fn_name);
        fprintf('[dry-run] %d iterations over: %s\n', total, strjoin(meta_keys, ', '));
        fprintf('[dry-run] inputs: %s\n', format_inputs(inputs, input_names));
        fprintf('[dry-run] outputs: {%s}\n', format_outputs(outputs));
        fprintf('\n');
    end

    % --- Compute Cartesian product ---
    if isempty(meta_values)
        combos = {{}};
    else
        combos = cartesian_product(meta_values);
    end

    completed = 0;
    skipped = 0;

    for c = 1:numel(combos)
        combo = combos{c};

        % Build metadata struct for this iteration
        metadata = struct();
        meta_nv = {};
        meta_parts = {};
        for k = 1:numel(meta_keys)
            val = combo{k};
            metadata.(meta_keys(k)) = val;
            meta_nv{end+1} = char(meta_keys(k)); %#ok<AGROW>
            meta_nv{end+1} = val; %#ok<AGROW>
            if isnumeric(val)
                meta_parts{end+1} = sprintf('%s=%g', meta_keys(k), val); %#ok<AGROW>
            else
                meta_parts{end+1} = sprintf('%s=%s', meta_keys(k), string(val)); %#ok<AGROW>
            end
        end
        metadata_str = strjoin(meta_parts, ', ');

        % --- Dry-run iteration ---
        if dry_run
            print_dry_run_iteration(inputs, input_names, outputs, ...
                metadata, meta_keys, metadata_str, should_pass_metadata);
            completed = completed + 1;
            continue;
        end

        % --- Load inputs ---
        loaded = cell(1, n_inputs);
        load_failed = false;

        for p = 1:n_inputs
            var_spec = inputs.(input_names{p});

            if isa(var_spec, 'scidb.Fixed')
                % Merge iteration metadata with fixed overrides
                load_nv = meta_nv;
                fixed_fields = fieldnames(var_spec.fixed_metadata);
                for f = 1:numel(fixed_fields)
                    load_nv{end+1} = fixed_fields{f}; %#ok<AGROW>
                    load_nv{end+1} = var_spec.fixed_metadata.(fixed_fields{f}); %#ok<AGROW>
                end
                var_inst = var_spec.var_type;
            else
                load_nv = meta_nv;
                var_inst = var_spec;
            end

            try
                loaded{p} = var_inst.load(load_nv{:});
            catch err
                fprintf('[skip] %s: failed to load %s (%s): %s\n', ...
                    metadata_str, input_names{p}, class(var_inst), err.message);
                load_failed = true;
                break;
            end
        end

        if load_failed
            skipped = skipped + 1;
            continue;
        end

        % --- Call the function ---
        fprintf('[run] %s: %s(%s)\n', metadata_str, fn_name, ...
            strjoin(string(input_names'), ', '));

        try
            % Determine how many outputs to capture.  For Thunks with
            % unpack_output=true MATLAB's subsref distributes outputs to
            % separate return values, so we must request them all.
            use_multi_out = isa(fn, 'scidb.Thunk') && fn.unpack_output ...
                && n_outputs > 1;

            if use_multi_out
                result = cell(1, n_outputs);
                if should_pass_metadata
                    [result{1:n_outputs}] = fn(loaded{:}, meta_nv{:});
                else
                    [result{1:n_outputs}] = fn(loaded{:});
                end
            else
                if should_pass_metadata
                    result = fn(loaded{:}, meta_nv{:});
                else
                    result = fn(loaded{:});
                end
                % Normalize single output to cell array
                if ~iscell(result)
                    result = {result};
                end
            end
        catch err
            fprintf('[skip] %s: %s raised: %s\n', ...
                metadata_str, fn_name, err.message);
            skipped = skipped + 1;
            continue;
        end

        % --- Save outputs ---
        if do_save
            for o = 1:min(n_outputs, numel(result))
                out_inst = outputs{o};
                try
                    out_inst.save(result{o}, meta_nv{:});
                    fprintf('[save] %s: %s\n', metadata_str, class(out_inst));
                catch err
                    fprintf('[error] %s: failed to save %s: %s\n', ...
                        metadata_str, class(out_inst), err.message);
                end
            end
        end

        completed = completed + 1;
    end

    % --- Summary ---
    fprintf('\n');
    if dry_run
        fprintf('[dry-run] would process %d iterations\n', total);
    else
        fprintf('[done] completed=%d, skipped=%d, total=%d\n', ...
            completed, skipped, total);
    end
end


% =========================================================================
% Local helper functions
% =========================================================================

function [meta_args, opts] = split_options(varargin)
%SPLIT_OPTIONS  Separate known option flags from metadata name-value pairs.
    opts.dry_run = false;
    opts.save = true;
    opts.pass_metadata = [];

    meta_args = {};
    i = 1;
    while i <= numel(varargin)
        key = varargin{i};
        if (ischar(key) || isstring(key))
            switch lower(string(key))
                case "dry_run"
                    opts.dry_run = logical(varargin{i+1});
                    i = i + 2;
                    continue;
                case "save"
                    opts.save = logical(varargin{i+1});
                    i = i + 2;
                    continue;
                case "pass_metadata"
                    opts.pass_metadata = logical(varargin{i+1});
                    i = i + 2;
                    continue;
            end
        end
        meta_args{end+1} = varargin{i}; %#ok<AGROW>
        i = i + 1;
    end
end


function s = format_inputs(inputs, input_names)
%FORMAT_INPUTS  Format the inputs struct for display.
    parts = cell(1, numel(input_names));
    for i = 1:numel(input_names)
        var_spec = inputs.(input_names{i});
        if isa(var_spec, 'scidb.Fixed')
            fields = fieldnames(var_spec.fixed_metadata);
            fixed_parts = cell(1, numel(fields));
            for f = 1:numel(fields)
                val = var_spec.fixed_metadata.(fields{f});
                if isnumeric(val)
                    fixed_parts{f} = sprintf('%s=%g', fields{f}, val);
                else
                    fixed_parts{f} = sprintf('%s=%s', fields{f}, string(val));
                end
            end
            parts{i} = sprintf('%s: Fixed(%s, %s)', input_names{i}, ...
                class(var_spec.var_type), strjoin(fixed_parts, ', '));
        else
            parts{i} = sprintf('%s: %s', input_names{i}, class(var_spec));
        end
    end
    s = ['{' strjoin(parts, ', ') '}'];
end


function s = format_outputs(outputs)
%FORMAT_OUTPUTS  Format the outputs cell array for display.
    parts = cell(1, numel(outputs));
    for i = 1:numel(outputs)
        parts{i} = class(outputs{i});
    end
    s = strjoin(parts, ', ');
end


function combos = cartesian_product(value_cells)
%CARTESIAN_PRODUCT  Compute Cartesian product of cell arrays.
    n = numel(value_cells);
    if n == 0
        combos = {{}};
        return;
    end

    sizes = cellfun(@numel, value_cells);
    idx_args = arrayfun(@(s) 1:s, sizes, 'UniformOutput', false);
    grids = cell(1, n);
    [grids{:}] = ndgrid(idx_args{:});

    total = prod(sizes);
    combos = cell(1, total);
    for t = 1:total
        combo = cell(1, n);
        for d = 1:n
            combo{d} = value_cells{d}{grids{d}(t)};
        end
        combos{t} = combo;
    end
end


function print_dry_run_iteration(inputs, input_names, outputs, ...
    metadata, meta_keys, metadata_str, pass_metadata)
%PRINT_DRY_RUN_ITERATION  Show what would happen for one iteration.
    fprintf('[dry-run] %s:\n', metadata_str);

    for p = 1:numel(input_names)
        var_spec = inputs.(input_names{p});
        if isa(var_spec, 'scidb.Fixed')
            load_meta = metadata;
            fields = fieldnames(var_spec.fixed_metadata);
            for f = 1:numel(fields)
                load_meta.(fields{f}) = var_spec.fixed_metadata.(fields{f});
            end
            type_name = class(var_spec.var_type);
        else
            load_meta = metadata;
            type_name = class(var_spec);
        end

        load_fields = fieldnames(load_meta);
        load_parts = cell(1, numel(load_fields));
        for f = 1:numel(load_fields)
            val = load_meta.(load_fields{f});
            if isnumeric(val)
                load_parts{f} = sprintf('%s=%g', load_fields{f}, val);
            else
                load_parts{f} = sprintf('%s=%s', load_fields{f}, string(val));
            end
        end
        fprintf('  load %s = %s().load(%s)\n', input_names{p}, ...
            type_name, strjoin(load_parts, ', '));
    end

    if pass_metadata
        fprintf('  pass metadata: %s\n', metadata_str);
    end

    for o = 1:numel(outputs)
        fprintf('  save %s().save(..., %s)\n', class(outputs{o}), metadata_str);
    end
end
