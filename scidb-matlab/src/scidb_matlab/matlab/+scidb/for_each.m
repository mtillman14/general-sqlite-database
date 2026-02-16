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
%   Inputs can be:
%   - BaseVariable instances — loaded from the database
%   - scidb.Fixed wrappers — loaded with overridden metadata
%   - scidb.PathInput instances — resolved to file paths
%   - Plain values (constants) — passed directly to the function and
%     included in the save metadata as version keys
%
%   Arguments:
%       fn      - Function handle or scidb.Thunk
%       inputs  - Struct mapping parameter names to BaseVariable instances,
%                 scidb.Fixed wrappers, or constant values.
%                 The field order determines argument order when calling fn.
%       outputs - Cell array of BaseVariable instances for output types
%
%   Name-Value Arguments:
%       dry_run       - If true, preview without executing (default: false)
%       save          - If true, save outputs (default: true)
%       preload       - If true, pre-load all input data for each variable
%                       type in a single query before iterating. Much faster
%                       but uses more memory. Set to false for very large
%                       datasets that may not fit in memory. (default: true)
%       pass_metadata - If true, pass metadata as trailing name-value
%                       arguments to fn. If not specified, auto-detects
%                       based on fn.generates_file when fn is a Thunk.
%       parallel      - If true, use 3-phase parallel execution:
%                       (1) serial pre-resolve, (2) parfor compute,
%                       (3) serial batch save. Requires pure MATLAB fn
%                       (no Thunks or PathInputs). With Parallel Computing
%                       Toolbox, parfor runs in parallel; without it,
%                       parfor silently runs serially. (default: false)
%       db            - Optional DatabaseManager to use for all load/save
%                       operations instead of the global database
%       (any other)   - Metadata iterables (numeric or string arrays)
%
%   Example:
%       scidb.for_each(@filter_data, ...
%           struct('step_length', StepLength(), ...
%                  'smoothing',   0.2), ...
%           {FilteredStepLength()}, ...
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
    do_preload = opts.preload;
    as_table_raw = opts.as_table;

    % Build db name-value pair for passthrough to load/save
    if isempty(opts.db)
        db_nv = {};
    else
        db_nv = {'db', opts.db};
    end

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

    % Resolve empty arrays to all distinct values from the database
    needs_resolve = false(1, numel(meta_keys));
    for i = 1:numel(meta_values)
        needs_resolve(i) = isempty(meta_values{i});
    end
    if any(needs_resolve)
        if isempty(opts.db)
            resolve_db = py.scidb.database.get_database();
        else
            resolve_db = opts.db;
        end
        for i = find(needs_resolve)
            py_vals = resolve_db.distinct_schema_values(char(meta_keys(i)));
            mat_vals = cell(py_vals);
            for j = 1:numel(mat_vals)
                mat_vals{j} = scidb.internal.from_python(mat_vals{j});
            end
            if isempty(mat_vals)
                fprintf('[warn] no values found for ''%s'' in database, 0 iterations\n', ...
                    meta_keys(i));
            end
            meta_values{i} = mat_vals;
        end
    end

    % Compute total iterations
    total = 1;
    for i = 1:numel(meta_values)
        total = total * numel(meta_values{i});
    end

    % Parse inputs struct — separate loadable inputs from constants
    input_names = fieldnames(inputs);
    n_inputs = numel(input_names);

    loadable_idx = false(1, n_inputs);
    constant_names = {};
    constant_values = {};
    constant_nv = {};  % name-value pairs for save metadata

    for p = 1:n_inputs
        var_spec = inputs.(input_names{p});
        if is_loadable(var_spec)
            loadable_idx(p) = true;
        else
            constant_names{end+1} = input_names{p}; %#ok<AGROW>
            constant_values{end+1} = var_spec; %#ok<AGROW>
            constant_nv{end+1} = input_names{p}; %#ok<AGROW>
            constant_nv{end+1} = var_spec; %#ok<AGROW>
        end
    end

    % Resolve as_table: true → all loadable input names, false/empty → none
    if islogical(as_table_raw) && isscalar(as_table_raw) && as_table_raw
        as_table_set = string(input_names(loadable_idx)');
    elseif islogical(as_table_raw) && isscalar(as_table_raw) && ~as_table_raw
        as_table_set = string.empty;
    else
        as_table_set = as_table_raw;
    end

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

    % --- Pre-load phase (optimization: 1 query per input instead of N) ---
    % Resolve database once for preloading and main loop
    if isempty(opts.db)
        py_db = py.scidb.database.get_database();
    else
        py_db = opts.db;
    end

    preloaded_results = cell(1, n_inputs);  % ThunkOutput arrays
    preloaded_maps    = cell(1, n_inputs);  % containers.Map: key_str → indices
    preloaded_keys    = cell(1, n_inputs);  % query key names per input

    if do_preload && ~dry_run && ~isempty(meta_keys)
        for p = 1:n_inputs
            if ~loadable_idx(p)
                continue;
            end

            var_spec = inputs.(input_names{p});

            % PathInput is not a database load — skip preloading
            if isa(var_spec, 'scidb.PathInput')
                continue;
            end

            % Determine variable type and fixed overrides
            if isa(var_spec, 'scidb.Fixed')
                var_inst = var_spec.var_type;
                fixed_meta = var_spec.fixed_metadata;
            else
                var_inst = var_spec;
                fixed_meta = struct();
            end

            type_name = class(var_inst);
            py_class = scidb.internal.ensure_registered(type_name);

            fprintf('Bulk preloading variable %s\n', type_name);

            % Build query metadata: iteration values as arrays + fixed overrides
            query_nv = {};
            for k = 1:numel(meta_keys)
                query_nv{end+1} = char(meta_keys(k)); %#ok<AGROW>
                vals = meta_values{k};
                if numel(vals) == 1
                    query_nv{end+1} = vals{1}; %#ok<AGROW>
                else
                    % Convert cell back to array for metadata_to_pydict
                    if isnumeric(vals{1})
                        query_nv{end+1} = cell2mat(vals); %#ok<AGROW>
                    else
                        query_nv{end+1} = string(vals); %#ok<AGROW>
                    end
                end
            end

            % Apply fixed overrides (scalar values replace array values)
            fixed_fields = fieldnames(fixed_meta);
            for f = 1:numel(fixed_fields)
                fld_name = fixed_fields{f};
                fval = fixed_meta.(fld_name);
                replaced = false;
                for k = 1:2:numel(query_nv)
                    if strcmp(query_nv{k}, fld_name)
                        query_nv{k+1} = fval;
                        replaced = true;
                        break;
                    end
                end
                if ~replaced
                    query_nv{end+1} = fld_name;   %#ok<AGROW>
                    query_nv{end+1} = fval;  %#ok<AGROW>
                end
            end

            % Track query keys for lookup (sorted)
            q_keys = string.empty;
            for k = 1:2:numel(query_nv)
                q_keys(end+1) = string(query_nv{k}); %#ok<AGROW>
            end
            preloaded_keys{p} = sort(q_keys);

            % Single query for all combinations — load_and_extract keeps
            % generator materialization in Python (no proxy overhead)
            py_metadata = scidb.internal.metadata_to_pydict(query_nv{:});
            bulk = py.scidb_matlab.bridge.load_and_extract( ...
                py_class, py_metadata, ...
                pyargs('version_id', 'latest', 'db', py_db));
            n_results = int64(bulk{'n'});

            if n_results == 0
                preloaded_results{p} = scidb.ThunkOutput.empty();
                preloaded_maps{p} = containers.Map();
                continue;
            end

            % Batch-wrap all results
            results = scidb.BaseVariable.wrap_py_vars_batch(bulk);
            preloaded_results{p} = results;

            % Build lookup map: metadata key string → array of indices
            lookup = containers.Map('KeyType', 'char', 'ValueType', 'any');
            for i = 1:numel(results)
                key_str = result_meta_key(results(i).metadata, preloaded_keys{p});
                if lookup.isKey(key_str)
                    lookup(key_str) = [lookup(key_str), i];
                else
                    lookup(key_str) = i;
                end
            end
            preloaded_maps{p} = lookup;
        end
    end

    % --- Parallel branch ---
    if opts.parallel && ~dry_run
        if isa(fn, 'scidb.Thunk')
            error('scidb:for_each', ...
                'parallel=true is not supported with Thunk functions (parfor workers cannot call Python).');
        end
        [completed, skipped] = run_parallel(fn, combos, n_inputs, n_outputs, ...
            input_names, loadable_idx, preloaded_results, preloaded_maps, ...
            preloaded_keys, inputs, meta_keys, outputs, ...
            constant_names, constant_values, constant_nv, ...
            as_table_set, should_pass_metadata, fn_name, do_save, db_nv, py_db);
        fprintf('\n[done] completed=%d, skipped=%d, total=%d\n', ...
            completed, skipped, total);
        return;
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

        % Build save metadata (iteration metadata + constants)
        save_nv = [meta_nv, constant_nv];

        % --- Dry-run iteration ---
        if dry_run
            print_dry_run_iteration(inputs, input_names, outputs, ...
                metadata, meta_keys, metadata_str, constant_nv, should_pass_metadata);
            completed = completed + 1;
            continue;
        end

        % --- Load inputs (only loadable ones, not constants) ---
        loaded = cell(1, n_inputs);
        load_failed = false;

        for p = 1:n_inputs
            if ~loadable_idx(p)
                % Constant — use value directly
                loaded{p} = inputs.(input_names{p});
                continue;
            end

            var_spec = inputs.(input_names{p});

            % Determine var_inst for table conversion
            if isa(var_spec, 'scidb.Fixed')
                var_inst = var_spec.var_type;
            else
                var_inst = var_spec;
            end

            % Use preloaded data if available for this input
            if ~isempty(preloaded_maps{p})
                fixed_meta = struct();
                if isa(var_spec, 'scidb.Fixed')
                    fixed_meta = var_spec.fixed_metadata;
                end
                key_str = combo_meta_key(meta_keys, combo, fixed_meta, preloaded_keys{p});

                if preloaded_maps{p}.isKey(key_str)
                    idx = preloaded_maps{p}(key_str);
                    if numel(idx) == 1
                        loaded{p} = preloaded_results{p}(idx);
                    else
                        loaded{p} = preloaded_results{p}(idx);
                    end
                else
                    fprintf('[skip] %s: no data found for %s (%s)\n', ...
                        metadata_str, input_names{p}, class(var_inst));
                    load_failed = true;
                    break;
                end
            else
                % Fallback: per-iteration load (PathInput, preload=false,
                % or no metadata keys)
                if isa(var_spec, 'scidb.Fixed')
                    load_nv = meta_nv;
                    fixed_fields = fieldnames(var_spec.fixed_metadata);
                    for f = 1:numel(fixed_fields)
                        load_nv{end+1} = fixed_fields{f}; %#ok<AGROW>
                        load_nv{end+1} = var_spec.fixed_metadata.(fixed_fields{f}); %#ok<AGROW>
                    end
                else
                    load_nv = meta_nv;
                end

                try
                    loaded{p} = var_inst.load(load_nv{:}, db_nv{:});
                catch err
                    fprintf('[skip] %s: failed to load %s (%s): %s\n', ...
                        metadata_str, input_names{p}, class(var_inst), err.message);
                    load_failed = true;
                    break;
                end
            end

            % Convert multi-result to table if requested
            if ~isempty(as_table_set) && ismember(string(input_names{p}), as_table_set) ...
                    && isa(loaded{p}, 'scidb.ThunkOutput') && numel(loaded{p}) > 1
                type_name = class(var_inst);
                loaded{p} = fe_multi_result_to_table(loaded{p}, type_name);
            end
        end

        if load_failed
            skipped = skipped + 1;
            continue;
        end

        % --- Call the function ---
        fprintf('[run] %s: %s(%s)\n', metadata_str, fn_name, ...
            strjoin(string(input_names'), ', '));

        % For plain function handles (not Thunks), unwrap ThunkOutput /
        % BaseVariable inputs to raw data so existing functions work
        % without modification.  Thunks handle their own unwrapping.
        % Only unwrap loadable inputs, not constants.
        if ~isa(fn, 'scidb.Thunk')
            for p = 1:n_inputs
                if loadable_idx(p) && ~istable(loaded{p})
                    loaded{p} = scidb.internal.unwrap_input(loaded{p});
                end
            end
        end

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

        % --- Save outputs (include constants in metadata) ---
        if do_save
            for o = 1:min(n_outputs, numel(result))
                out_inst = outputs{o};
                try
                    out_inst.save(result{o}, save_nv{:}, db_nv{:});
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
% Parallel execution (3-phase: pre-resolve → parfor → batch save)
% =========================================================================

function [completed, skipped] = run_parallel(fn, combos, n_inputs, n_outputs, ...
    input_names, loadable_idx, preloaded_results, preloaded_maps, ...
    preloaded_keys, inputs, meta_keys, outputs, ...
    constant_names, constant_values, constant_nv, ...
    as_table_set, should_pass_metadata, fn_name, do_save, db_nv, py_db)
%RUN_PARALLEL  Three-phase parallel execution for for_each.
%   Phase A: pre-resolve all inputs from preloaded maps (serial, uses py.)
%   Phase B: parfor compute (pure MATLAB, no py. calls)
%   Phase C: batch save results (serial, uses py.)

    n_combos = numel(combos);

    % Pre-allocate per-combo storage
    all_inputs = cell(1, n_combos);    % each: cell array of fn arguments
    all_meta_nv = cell(1, n_combos);   % each: cell array of meta name-value pairs
    all_save_nv = cell(1, n_combos);   % each: cell array of save name-value pairs
    resolve_ok = false(1, n_combos);

    % ---- Phase A: Pre-resolve all inputs (serial) ----
    fprintf('[parallel] Phase A: pre-resolving %d combinations...\n', n_combos);

    for c = 1:n_combos
        combo = combos{c};

        % Build metadata name-value pairs for this combo
        meta_nv = {};
        meta_parts = {};
        for k = 1:numel(meta_keys)
            val = combo{k};
            meta_nv{end+1} = char(meta_keys(k)); %#ok<AGROW>
            meta_nv{end+1} = val; %#ok<AGROW>
            if isnumeric(val)
                meta_parts{end+1} = sprintf('%s=%g', meta_keys(k), val); %#ok<AGROW>
            else
                meta_parts{end+1} = sprintf('%s=%s', meta_keys(k), string(val)); %#ok<AGROW>
            end
        end
        metadata_str = strjoin(meta_parts, ', ');
        all_meta_nv{c} = meta_nv;
        all_save_nv{c} = [meta_nv, constant_nv];

        % Resolve each input
        loaded = cell(1, n_inputs);
        load_failed = false;

        for p = 1:n_inputs
            if ~loadable_idx(p)
                % Constant — use value directly
                loaded{p} = inputs.(input_names{p});
                continue;
            end

            var_spec = inputs.(input_names{p});

            % PathInput not supported in parallel mode
            if isa(var_spec, 'scidb.PathInput')
                error('scidb:for_each', ...
                    'parallel=true is not supported with PathInput (path resolution may need Python).');
            end

            % Determine var_inst for table conversion
            if isa(var_spec, 'scidb.Fixed')
                var_inst = var_spec.var_type;
            else
                var_inst = var_spec;
            end

            % Use preloaded data
            if ~isempty(preloaded_maps{p})
                fixed_meta = struct();
                if isa(var_spec, 'scidb.Fixed')
                    fixed_meta = var_spec.fixed_metadata;
                end
                key_str = combo_meta_key(meta_keys, combo, fixed_meta, preloaded_keys{p});

                if preloaded_maps{p}.isKey(key_str)
                    idx = preloaded_maps{p}(key_str);
                    if numel(idx) == 1
                        loaded{p} = preloaded_results{p}(idx);
                    else
                        loaded{p} = preloaded_results{p}(idx);
                    end
                else
                    fprintf('[skip] %s: no data found for %s (%s)\n', ...
                        metadata_str, input_names{p}, class(var_inst));
                    load_failed = true;
                    break;
                end
            else
                % Fallback: per-iteration load (preload=false or no metadata)
                if isa(var_spec, 'scidb.Fixed')
                    load_nv = meta_nv;
                    fixed_fields = fieldnames(var_spec.fixed_metadata);
                    for f = 1:numel(fixed_fields)
                        load_nv{end+1} = fixed_fields{f}; %#ok<AGROW>
                        load_nv{end+1} = var_spec.fixed_metadata.(fixed_fields{f}); %#ok<AGROW>
                    end
                else
                    load_nv = meta_nv;
                end
                try
                    loaded{p} = var_inst.load(load_nv{:}, db_nv{:});
                catch err
                    fprintf('[skip] %s: failed to load %s (%s): %s\n', ...
                        metadata_str, input_names{p}, class(var_inst), err.message);
                    load_failed = true;
                    break;
                end
            end

            % Convert multi-result to table if requested
            if ~isempty(as_table_set) && ismember(string(input_names{p}), as_table_set) ...
                    && isa(loaded{p}, 'scidb.ThunkOutput') && numel(loaded{p}) > 1
                type_name = class(var_inst);
                loaded{p} = fe_multi_result_to_table(loaded{p}, type_name);
            end
        end

        if load_failed
            continue;
        end

        % Unwrap ThunkOutput/BaseVariable inputs to raw data (same as serial path)
        for p = 1:n_inputs
            if loadable_idx(p) && ~istable(loaded{p})
                loaded{p} = scidb.internal.unwrap_input(loaded{p});
            end
        end

        all_inputs{c} = loaded;
        resolve_ok(c) = true;
    end

    n_resolved = sum(resolve_ok);
    fprintf('[parallel] Phase A done: %d resolved, %d skipped\n', ...
        n_resolved, n_combos - n_resolved);

    % ---- Phase B: Parallel compute (parfor) ----
    fprintf('[parallel] Phase B: computing %d items with parfor...\n', n_resolved);

    resolved_indices = find(resolve_ok);
    % Copy into contiguous arrays for parfor (avoid broadcast of sparse cells)
    par_inputs = cell(1, n_resolved);
    par_meta_nv = cell(1, n_resolved);
    for j = 1:n_resolved
        par_inputs{j} = all_inputs{resolved_indices(j)};
        par_meta_nv{j} = all_meta_nv{resolved_indices(j)};
    end

    results = cell(1, n_resolved);
    compute_ok = true(1, n_resolved);
    compute_errors = cell(1, n_resolved);

    parfor j = 1:n_resolved
        try
            if should_pass_metadata
                r = fn(par_inputs{j}{:}, par_meta_nv{j}{:});
            else
                r = fn(par_inputs{j}{:});
            end
            if ~iscell(r)
                r = {r};
            end
            results{j} = r;
        catch err
            compute_ok(j) = false;
            compute_errors{j} = err.message;
            results{j} = {};
        end
    end

    % Report compute errors
    for j = find(~compute_ok)
        c = resolved_indices(j);
        combo = combos{c};
        meta_parts = {};
        for k = 1:numel(meta_keys)
            val = combo{k};
            if isnumeric(val)
                meta_parts{end+1} = sprintf('%s=%g', meta_keys(k), val); %#ok<AGROW>
            else
                meta_parts{end+1} = sprintf('%s=%s', meta_keys(k), string(val)); %#ok<AGROW>
            end
        end
        fprintf('[skip] %s: %s raised: %s\n', ...
            strjoin(meta_parts, ', '), fn_name, compute_errors{j});
    end

    n_computed = sum(compute_ok);
    fprintf('[parallel] Phase B done: %d succeeded, %d failed\n', ...
        n_computed, n_resolved - n_computed);

    % ---- Phase C: Batch save (serial) ----
    if do_save && n_computed > 0
        fprintf('[parallel] Phase C: batch saving %d results...\n', n_computed);

        for o = 1:n_outputs
            type_name = class(outputs{o});
            scidb.internal.ensure_registered(type_name);

            py_data = py.list();
            py_metas = py.list();
            save_count = 0;

            for j = find(compute_ok)
                c = resolved_indices(j);
                if o <= numel(results{j})
                    py_data.append(scidb.internal.to_python(results{j}{o}));
                    py_metas.append(scidb.internal.metadata_to_pydict(all_save_nv{c}{:}));
                    save_count = save_count + 1;
                end
            end

            if save_count > 0
                py.scidb_matlab.bridge.for_each_batch_save( ...
                    type_name, py_data, py_metas, py_db);
                fprintf('[save] %s: %d items (batch)\n', type_name, save_count);
            end
        end
    end

    completed = n_computed;
    skipped = numel(combos) - n_computed;
end


% =========================================================================
% Local helper functions
% =========================================================================

function tf = is_loadable(var_spec)
%IS_LOADABLE  Check if an input spec is a loadable type.
%   Returns true for BaseVariable instances, Fixed wrappers, and PathInput.
%   Returns false for plain constants (numeric, string, logical, etc.).
    tf = isa(var_spec, 'scidb.BaseVariable') ...
      || isa(var_spec, 'scidb.Fixed') ...
      || isa(var_spec, 'scidb.PathInput');
end


function [meta_args, opts] = split_options(varargin)
%SPLIT_OPTIONS  Separate known option flags from metadata name-value pairs.
    opts.dry_run = false;
    opts.save = true;
    opts.preload = true;
    opts.pass_metadata = [];
    opts.as_table = string.empty;
    opts.db = [];
    opts.parallel = false;

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
                case "preload"
                    opts.preload = logical(varargin{i+1});
                    i = i + 2;
                    continue;
                case "pass_metadata"
                    opts.pass_metadata = logical(varargin{i+1});
                    i = i + 2;
                    continue;
                case "as_table"
                    val = varargin{i+1};
                    if islogical(val)
                        opts.as_table = val;
                    elseif isstring(val)
                        opts.as_table = val;
                    elseif ischar(val)
                        opts.as_table = string(val);
                    elseif iscell(val)
                        opts.as_table = string(val);
                    end
                    i = i + 2;
                    continue;
                case "db"
                    opts.db = varargin{i+1};
                    i = i + 2;
                    continue;
                case "parallel"
                    opts.parallel = logical(varargin{i+1});
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
        elseif is_loadable(var_spec)
            parts{i} = sprintf('%s: %s', input_names{i}, class(var_spec));
        else
            parts{i} = sprintf('%s: %s', input_names{i}, format_value(var_spec));
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


function s = format_value(val)
%FORMAT_VALUE  Format a constant value for display.
    if isnumeric(val)
        s = sprintf('%g', val);
    elseif islogical(val)
        if val
            s = 'true';
        else
            s = 'false';
        end
    elseif ischar(val) || isstring(val)
        s = sprintf('''%s''', string(val));
    else
        s = mat2str(val);
    end
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
    metadata, meta_keys, metadata_str, constant_nv, pass_metadata)
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
        elseif is_loadable(var_spec)
            type_name = class(var_spec);

            load_fields = fieldnames(metadata);
            load_parts = cell(1, numel(load_fields));
            for f = 1:numel(load_fields)
                val = metadata.(load_fields{f});
                if isnumeric(val)
                    load_parts{f} = sprintf('%s=%g', load_fields{f}, val);
                else
                    load_parts{f} = sprintf('%s=%s', load_fields{f}, string(val));
                end
            end
            fprintf('  load %s = %s().load(%s)\n', input_names{p}, ...
                type_name, strjoin(load_parts, ', '));
        else
            fprintf('  constant %s = %s\n', input_names{p}, ...
                format_value(var_spec));
        end
    end

    if pass_metadata
        fprintf('  pass metadata: %s\n', metadata_str);
    end

    % Build save metadata string (iteration metadata + constants)
    save_parts = {};
    load_fields = fieldnames(metadata);
    for f = 1:numel(load_fields)
        val = metadata.(load_fields{f});
        if isnumeric(val)
            save_parts{end+1} = sprintf('%s=%g', load_fields{f}, val); %#ok<AGROW>
        else
            save_parts{end+1} = sprintf('%s=%s', load_fields{f}, string(val)); %#ok<AGROW>
        end
    end
    for i = 1:2:numel(constant_nv)
        save_parts{end+1} = sprintf('%s=%s', constant_nv{i}, ...
            format_value(constant_nv{i+1})); %#ok<AGROW>
    end
    save_metadata_str = strjoin(save_parts, ', ');

    for o = 1:numel(outputs)
        fprintf('  save %s().save(..., %s)\n', class(outputs{o}), save_metadata_str);
    end
end


function tbl = fe_multi_result_to_table(results, type_name)
%FE_MULTI_RESULT_TO_TABLE  Convert an array of ThunkOutput to a MATLAB table.
    n = numel(results);

    % Build metadata columns
    meta_fields = fieldnames(results(1).metadata);
    tbl = table();
    for f = 1:numel(meta_fields)
        col_data = cell(n, 1);
        for i = 1:n
            if isfield(results(i).metadata, meta_fields{f})
                col_data{i} = results(i).metadata.(meta_fields{f});
            else
                col_data{i} = missing;
            end
        end
        tbl.(meta_fields{f}) = col_data;
    end

    % version_id column
    vid_data = zeros(n, 1);
    for i = 1:n
        if ~isempty(results(i).version_id)
            vid_data(i) = results(i).version_id;
        end
    end
    tbl.version_id = vid_data;

    % Data column (named after the variable type)
    parts = strsplit(type_name, '.');
    col_name = parts{end};
    data_col = cell(n, 1);
    for i = 1:n
        data_col{i} = results(i).data;
    end
    tbl.(col_name) = data_col;
end


function key = build_meta_key(keys, vals)
%BUILD_META_KEY  Build a sorted lookup key from metadata key-value pairs.
%   keys: string array, vals: cell array of corresponding values.
%   Returns a string like "session=A|subject=1" (sorted by key name).
    parts = cell(1, numel(keys));
    for k = 1:numel(keys)
        v = vals{k};
        if isnumeric(v)
            parts{k} = sprintf('%s=%g', keys(k), v);
        else
            parts{k} = sprintf('%s=%s', keys(k), string(v));
        end
    end
    key = char(strjoin(sort(string(parts)), '|'));
end


function key = result_meta_key(metadata_struct, query_keys)
%RESULT_META_KEY  Build lookup key from a loaded result's metadata struct.
    vals = cell(1, numel(query_keys));
    for k = 1:numel(query_keys)
        vals{k} = metadata_struct.(char(query_keys(k)));
    end
    key = build_meta_key(query_keys, vals);
end


function key = combo_meta_key(meta_keys, combo, fixed_meta, query_keys)
%COMBO_META_KEY  Build lookup key for a specific iteration combo.
%   Applies fixed metadata overrides and uses only query_keys for the key.
    % Start with iteration metadata
    effective = containers.Map('KeyType', 'char', 'ValueType', 'any');
    for k = 1:numel(meta_keys)
        effective(char(meta_keys(k))) = combo{k};
    end

    % Apply fixed overrides
    ff = fieldnames(fixed_meta);
    for f = 1:numel(ff)
        effective(ff{f}) = fixed_meta.(ff{f});
    end

    % Extract values for query_keys only
    vals = cell(1, numel(query_keys));
    for k = 1:numel(query_keys)
        vals{k} = effective(char(query_keys(k)));
    end
    key = build_meta_key(query_keys, vals);
end
