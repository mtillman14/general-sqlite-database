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
%       distribute    - If true, split each output (vector/table) by
%                       element/row and save each piece at the schema level
%                       immediately below the deepest iterated key, using
%                       1-based indexing. For example, with schema
%                       [subject, trial, cycle] and iteration at the trial
%                       level, distribute=true saves each element/row as a
%                       separate cycle (1, 2, 3, ...). (default: false)
%       db            - Optional DatabaseManager to use for all load/save
%                       operations instead of the global database
%       where         - Optional scidb.Filter to apply when loading each input
%                       variable. Combos where an input has no matching data
%                       after filtering are silently skipped.
%                       Example: where=Side() == "R"
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

    % Build where name-value pair for passthrough to load calls
    where_filter = opts.where;
    if isempty(where_filter)
        where_nv = {};
    else
        where_nv = {'where', where_filter};
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

    % Validate distribute parameter and resolve target key
    distribute = opts.distribute;
    distribute_key = '';
    if distribute
        if isempty(opts.db)
            dist_db = py.scidb.database.get_database();
        else
            dist_db = opts.db;
        end
        schema_keys = cell(dist_db.dataset_schema_keys);
        for sk = 1:numel(schema_keys)
            schema_keys{sk} = string(schema_keys{sk});
        end
        schema_keys = [schema_keys{:}];

        iter_keys_in_schema = schema_keys(ismember(schema_keys, meta_keys));
        if isempty(iter_keys_in_schema)
            error('scidb:for_each', ...
                'distribute=true requires at least one metadata_iterable that is a schema key.');
        end
        deepest_iterated = iter_keys_in_schema(end);
        deepest_idx = find(schema_keys == deepest_iterated, 1);

        if deepest_idx >= numel(schema_keys)
            error('scidb:for_each', ...
                'distribute=true but ''%s'' is the deepest schema key. There is no lower level to distribute to. Schema order: %s', ...
                deepest_iterated, strjoin(schema_keys, ', '));
        end
        distribute_key = schema_keys(deepest_idx + 1);
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
            % Only include in save metadata if value can be used as a
            % metadata key (scalar numeric/string/char/logical, or struct).
            % Tables, matrices, cell arrays are data inputs, not parameters.
            if is_metadata_compatible(var_spec)
                constant_nv{end+1} = input_names{p}; %#ok<AGROW>
                constant_nv{end+1} = var_spec; %#ok<AGROW>
            end
        end
    end

    % Check distribute doesn't conflict with a constant input name
    if strlength(distribute_key) > 0 && ismember(distribute_key, string(constant_names))
        error('scidb:for_each', ...
            'distribute target ''%s'' conflicts with a constant input named ''%s''.', ...
            distribute_key, distribute_key);
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
        if ~isempty(where_filter)
            fprintf('[dry-run] where: filter applied to all input loads\n');
        end
        if strlength(distribute_key) > 0
            fprintf('[dry-run] distribute: ''%s'' (split outputs by element/row, 1-based)\n', distribute_key);
        end
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

            % Merge loads constituents individually — skip preloading
            if isa(var_spec, 'scidb.Merge')
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
            if isempty(where_filter)
                bulk = py.scidb_matlab.bridge.load_and_extract( ...
                    py_class, py_metadata, ...
                    pyargs('version_id', 'latest', 'db', py_db));
            else
                bulk = py.scidb_matlab.bridge.load_and_extract( ...
                    py_class, py_metadata, ...
                    pyargs('version_id', 'latest', 'db', py_db, ...
                           'where', where_filter.py_filter));
            end
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
            as_table_set, should_pass_metadata, fn_name, do_save, db_nv, py_db, ...
            where_nv);
        fprintf('\n[done] completed=%d, skipped=%d, total=%d\n', ...
            completed, skipped, total);
        return;
    end

    completed = 0;
    skipped = 0;

    % Batch save: accumulate data+metadata in py.list objects, flush after
    % the loop via for_each_batch_save.  Thunks have lineage that requires
    % the per-record save path, so batch save is only for plain functions.
    use_batch_save = do_save && ~isa(fn, 'scidb.Thunk');

    if use_batch_save
        batch_accum = cell(1, n_outputs);
        for o = 1:n_outputs
            batch_accum{o}.py_data = py.list();
            batch_accum{o}.py_metas = py.list();
            batch_accum{o}.count = 0;
        end
    end

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
                metadata, meta_keys, metadata_str, constant_nv, should_pass_metadata, distribute_key);
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

            % Handle Merge: load each constituent and combine into table
            if isa(var_spec, 'scidb.Merge')
                try
                    loaded{p} = merge_constituents(var_spec, meta_nv, db_nv, input_names{p});
                catch err
                    fprintf('[skip] %s: failed to merge %s: %s\n', ...
                        metadata_str, input_names{p}, err.message);
                    load_failed = true;
                    break;
                end
                continue;
            end

            % Guard against Fixed wrapping Merge
            if isa(var_spec, 'scidb.Fixed') && isa(var_spec.var_type, 'scidb.Merge')
                error('scidb:for_each', ...
                    'Fixed cannot wrap a Merge. Use Fixed on individual constituents inside the Merge instead: Merge(Fixed(VarA(), ...), VarB())');
            end

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
                    loaded{p} = var_inst.load(load_nv{:}, db_nv{:}, where_nv{:});
                catch err
                    fprintf('[skip] %s: failed to load %s (%s): %s\n', ...
                        metadata_str, input_names{p}, class(var_inst), err.message);
                    load_failed = true;
                    break;
                end
            end

            % Handle as_table conversion and/or column selection
            has_col_sel = isa(var_inst, 'scidb.BaseVariable') && ~isempty(var_inst.selected_columns);
            wants_table = ~isempty(as_table_set) && ismember(string(input_names{p}), as_table_set) ...
                    && isa(loaded{p}, 'scidb.ThunkOutput') && numel(loaded{p}) > 1;

            if has_col_sel && wants_table
                % Both active: filter each variable's data BEFORE building
                % the table so metadata columns are preserved
                cols = var_inst.selected_columns;
                try
                    for qi = 1:numel(loaded{p})
                        tbl_i = loaded{p}(qi).data;
                        if ~istable(tbl_i)
                            error('scidb:for_each', ...
                                'Column selection on ''%s'' requires table data, but loaded data is %s.', ...
                                input_names{p}, class(tbl_i));
                        end
                        for ci = 1:numel(cols)
                            if ~ismember(cols(ci), tbl_i.Properties.VariableNames)
                                error('scidb:for_each', ...
                                    'Column ''%s'' not found in ''%s''. Available columns: %s', ...
                                    cols(ci), input_names{p}, strjoin(tbl_i.Properties.VariableNames, ', '));
                            end
                        end
                        loaded{p}(qi).data = tbl_i(:, cols);
                    end
                catch err
                    fprintf('[skip] %s: column selection failed for %s: %s\n', ...
                        metadata_str, input_names{p}, err.message);
                    load_failed = true;
                    break;
                end
                type_name = class(var_inst);
                loaded{p} = fe_multi_result_to_table(loaded{p}, type_name);
            elseif wants_table
                type_name = class(var_inst);
                loaded{p} = fe_multi_result_to_table(loaded{p}, type_name);
            elseif has_col_sel
                try
                    loaded{p} = apply_column_selection(loaded{p}, var_inst.selected_columns, input_names{p});
                catch err
                    fprintf('[skip] %s: column selection failed for %s: %s\n', ...
                        metadata_str, input_names{p}, err.message);
                    load_failed = true;
                    break;
                end
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
            if strlength(distribute_key) > 0
                % Distribute mode: split table output by row, saving each row as a
                % separate record under a distinct value of distribute_key.
                % If distribute_key already exists as a column, use those
                % values; otherwise assign 1-based row indices.
                % For non-table outputs, split by element/row.
                for o = 1:min(n_outputs, numel(result))
                    out_inst = outputs{o};
                    raw_value = result{o};
                    % Unwrap ThunkOutput/BaseVariable to raw data
                    if isa(raw_value, 'scidb.ThunkOutput') || isa(raw_value, 'scidb.BaseVariable')
                        raw_value = raw_value.data;
                    end

                    try
                        dist_key_char = char(distribute_key);
                        if istable(raw_value)
                            % Determine per-row distribute-key values
                            if ismember(dist_key_char, raw_value.Properties.VariableNames)
                                dist_values = raw_value.(dist_key_char);
                                data_tbl = raw_value;
                                data_tbl.(dist_key_char) = [];  % remove from data
                            else
                                dist_values = (1:height(raw_value))';
                                data_tbl = raw_value;
                            end
                            % Save each row as an individual record
                            for rowIdx = 1:height(data_tbl)
                                row_data = data_tbl(rowIdx, :);
                                dist_val = dist_values(rowIdx);
                                if isnumeric(dist_val) && isscalar(dist_val)
                                    save_meta = [meta_nv, constant_nv, {dist_key_char, dist_val}];
                                else
                                    save_meta = [meta_nv, constant_nv, {dist_key_char, char(string(dist_val))}];
                                end
                                if use_batch_save
                                    batch_accum{o}.py_data.append(scidb.internal.to_python(row_data));
                                    batch_accum{o}.py_metas.append(scidb.internal.metadata_to_pydict(save_meta{:}));
                                    batch_accum{o}.count = batch_accum{o}.count + 1;
                                else
                                    out_inst.save(row_data, save_meta{:}, db_nv{:});
                                    fprintf('[save] %s, %s=%s: %s (distribute row %d)\n', ...
                                        metadata_str, dist_key_char, num2str(dist_val), class(out_inst), rowIdx);
                                end
                            end
                        else
                            % Non-table: split by element/row and save each piece
                            pieces = split_for_distribute(raw_value);
                            for k = 1:numel(pieces)
                                save_meta = [meta_nv, constant_nv, {dist_key_char, k}];
                                if use_batch_save
                                    batch_accum{o}.py_data.append(scidb.internal.to_python(pieces{k}));
                                    batch_accum{o}.py_metas.append(scidb.internal.metadata_to_pydict(save_meta{:}));
                                    batch_accum{o}.count = batch_accum{o}.count + 1;
                                else
                                    out_inst.save(pieces{k}, save_meta{:}, db_nv{:});
                                end
                            end
                            if ~use_batch_save
                                fprintf('[save] %s: distributed %d pieces by ''%s''\n', ...
                                    metadata_str, numel(pieces), distribute_key);
                            end
                        end
                    catch err
                        fprintf('[error] %s: cannot distribute %s: %s\n', ...
                            metadata_str, class(out_inst), err.message);
                        continue;
                    end
                end
            else
                % Normal mode: save each output directly
                for o = 1:min(n_outputs, numel(result))
                    out_inst = outputs{o};
                    if use_batch_save
                        try
                            raw_value = result{o};
                            if isa(raw_value, 'scidb.ThunkOutput') || isa(raw_value, 'scidb.BaseVariable')
                                raw_value = raw_value.data;
                            end
                            batch_accum{o}.py_data.append(scidb.internal.to_python(raw_value));
                            batch_accum{o}.py_metas.append(scidb.internal.metadata_to_pydict(save_nv{:}));
                            batch_accum{o}.count = batch_accum{o}.count + 1;
                        catch err
                            fprintf('[error] %s: failed to convert %s for batch: %s\n', ...
                                metadata_str, class(out_inst), err.message);
                        end
                    else
                        try
                            out_inst.save(result{o}, save_nv{:}, db_nv{:});
                            fprintf('[save] %s: %s\n', metadata_str, class(out_inst));
                        catch err
                            fprintf('[error] %s: failed to save %s: %s\n', ...
                                metadata_str, class(out_inst), err.message);
                        end
                    end
                end
            end
        end

        completed = completed + 1;
    end

    % --- Batch save flush ---
    if use_batch_save && ~dry_run
        for o = 1:n_outputs
            if batch_accum{o}.count > 0
                type_name = class(outputs{o});
                scidb.internal.ensure_registered(type_name);
                py.scidb_matlab.bridge.for_each_batch_save( ...
                    type_name, batch_accum{o}.py_data, ...
                    batch_accum{o}.py_metas, py_db);
                fprintf('[save] %s: %d items (batch)\n', type_name, batch_accum{o}.count);
            end
        end
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
    as_table_set, should_pass_metadata, fn_name, do_save, db_nv, py_db, ...
    where_nv)
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
                    loaded{p} = var_inst.load(load_nv{:}, db_nv{:}, where_nv{:});
                catch err
                    fprintf('[skip] %s: failed to load %s (%s): %s\n', ...
                        metadata_str, input_names{p}, class(var_inst), err.message);
                    load_failed = true;
                    break;
                end
            end

            % Handle as_table conversion and/or column selection
            has_col_sel = isa(var_inst, 'scidb.BaseVariable') && ~isempty(var_inst.selected_columns);
            wants_table = ~isempty(as_table_set) && ismember(string(input_names{p}), as_table_set) ...
                    && isa(loaded{p}, 'scidb.ThunkOutput') && numel(loaded{p}) > 1;

            if has_col_sel && wants_table
                % Both active: filter each variable's data BEFORE building
                % the table so metadata columns are preserved
                cols = var_inst.selected_columns;
                try
                    for qi = 1:numel(loaded{p})
                        tbl_i = loaded{p}(qi).data;
                        if ~istable(tbl_i)
                            error('scidb:for_each', ...
                                'Column selection on ''%s'' requires table data, but loaded data is %s.', ...
                                input_names{p}, class(tbl_i));
                        end
                        for ci = 1:numel(cols)
                            if ~ismember(cols(ci), tbl_i.Properties.VariableNames)
                                error('scidb:for_each', ...
                                    'Column ''%s'' not found in ''%s''. Available columns: %s', ...
                                    cols(ci), input_names{p}, strjoin(tbl_i.Properties.VariableNames, ', '));
                            end
                        end
                        loaded{p}(qi).data = tbl_i(:, cols);
                    end
                catch err
                    fprintf('[skip] %s: column selection failed for %s: %s\n', ...
                        metadata_str, input_names{p}, err.message);
                    load_failed = true;
                    break;
                end
                type_name = class(var_inst);
                loaded{p} = fe_multi_result_to_table(loaded{p}, type_name);
            elseif wants_table
                type_name = class(var_inst);
                loaded{p} = fe_multi_result_to_table(loaded{p}, type_name);
            elseif has_col_sel
                try
                    loaded{p} = apply_column_selection(loaded{p}, var_inst.selected_columns, input_names{p});
                catch err
                    fprintf('[skip] %s: column selection failed for %s: %s\n', ...
                        metadata_str, input_names{p}, err.message);
                    load_failed = true;
                    break;
                end
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
%   Returns true for BaseVariable instances, Fixed wrappers, PathInput,
%   and Merge wrappers.
%   Returns false for plain constants (numeric, string, logical, etc.).
    tf = isa(var_spec, 'scidb.BaseVariable') ...
      || isa(var_spec, 'scidb.Fixed') ...
      || isa(var_spec, 'scidb.PathInput') ...
      || isa(var_spec, 'scidb.Merge');
end


function tf = is_metadata_compatible(val)
%IS_METADATA_COMPATIBLE  Return true if val can be used as a save metadata key.
%   Only scalar numeric/logical, scalar string/char, and structs are
%   compatible with Python metadata serialization.  Tables, matrices,
%   cell arrays, and other complex types cannot be stored as version keys.
    tf = (isnumeric(val) && isscalar(val)) ...
      || (islogical(val) && isscalar(val)) ...
      || (isstring(val) && isscalar(val)) ...
      || ischar(val) ...
      || isstruct(val);
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
    opts.distribute = false;
    opts.where = [];

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
                case "distribute"
                    opts.distribute = logical(varargin{i+1});
                    i = i + 2;
                    continue;
                case "where"
                    opts.where = varargin{i+1};
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
        if isa(var_spec, 'scidb.Merge')
            sub_parts = cell(1, numel(var_spec.var_specs));
            for j = 1:numel(var_spec.var_specs)
                sub = var_spec.var_specs{j};
                if isa(sub, 'scidb.Fixed')
                    sub_parts{j} = sprintf('Fixed(%s)', class(sub.var_type));
                else
                    sub_parts{j} = class(sub);
                end
            end
            parts{i} = sprintf('%s: Merge(%s)', input_names{i}, strjoin(sub_parts, ', '));
        elseif isa(var_spec, 'scidb.Fixed')
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
            if isa(var_spec, 'scidb.BaseVariable') && ~isempty(var_spec.selected_columns)
                col_str = strjoin(var_spec.selected_columns, '", "');
                parts{i} = sprintf('%s: %s["%s"]', input_names{i}, class(var_spec), col_str);
            else
                parts{i} = sprintf('%s: %s', input_names{i}, class(var_spec));
            end
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
    elseif istable(val)
        s = sprintf('<table %dx%d>', height(val), width(val));
    else
        try
            s = mat2str(val);
        catch
            s = sprintf('<%s>', class(val));
        end
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
    metadata, meta_keys, metadata_str, constant_nv, pass_metadata, distribute)
%PRINT_DRY_RUN_ITERATION  Show what would happen for one iteration.
    fprintf('[dry-run] %s:\n', metadata_str);

    for p = 1:numel(input_names)
        var_spec = inputs.(input_names{p});
        if isa(var_spec, 'scidb.Merge')
            fprintf('  merge %s:\n', input_names{p});
            for mi = 1:numel(var_spec.var_specs)
                sub = var_spec.var_specs{mi};
                if isa(sub, 'scidb.Fixed')
                    sub_meta = metadata;
                    ff = fieldnames(sub.fixed_metadata);
                    for fi = 1:numel(ff)
                        sub_meta.(ff{fi}) = sub.fixed_metadata.(ff{fi});
                    end
                    sub_fields = fieldnames(sub_meta);
                    sp = cell(1, numel(sub_fields));
                    for fi = 1:numel(sub_fields)
                        val = sub_meta.(sub_fields{fi});
                        if isnumeric(val)
                            sp{fi} = sprintf('%s=%g', sub_fields{fi}, val);
                        else
                            sp{fi} = sprintf('%s=%s', sub_fields{fi}, string(val));
                        end
                    end
                    fprintf('    [%d] %s().load(%s)\n', mi-1, class(sub.var_type), strjoin(sp, ', '));
                elseif isa(sub, 'scidb.BaseVariable')
                    sub_fields = fieldnames(metadata);
                    sp = cell(1, numel(sub_fields));
                    for fi = 1:numel(sub_fields)
                        val = metadata.(sub_fields{fi});
                        if isnumeric(val)
                            sp{fi} = sprintf('%s=%g', sub_fields{fi}, val);
                        else
                            sp{fi} = sprintf('%s=%s', sub_fields{fi}, string(val));
                        end
                    end
                    if ~isempty(sub.selected_columns)
                        col_str = strjoin(sub.selected_columns, ', ');
                        fprintf('    [%d] %s().load(%s) -> columns: [%s]\n', mi-1, class(sub), strjoin(sp, ', '), col_str);
                    else
                        fprintf('    [%d] %s().load(%s)\n', mi-1, class(sub), strjoin(sp, ', '));
                    end
                end
            end
        elseif isa(var_spec, 'scidb.Fixed')
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
            if ~isempty(var_spec.selected_columns)
                col_str = strjoin(var_spec.selected_columns, ', ');
                fprintf('  load %s = %s().load(%s) -> columns: [%s]\n', input_names{p}, ...
                    type_name, strjoin(load_parts, ', '), col_str);
            else
                fprintf('  load %s = %s().load(%s)\n', input_names{p}, ...
                    type_name, strjoin(load_parts, ', '));
            end
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
        if strlength(distribute) > 0
            fprintf('  distribute %s by ''%s'' (1-based indexing)\n', class(outputs{o}), distribute);
        else
            fprintf('  save %s().save(..., %s)\n', class(outputs{o}), save_metadata_str);
        end
    end
end


function pieces = split_for_distribute(data)
%SPLIT_FOR_DISTRIBUTE  Split data into elements for distribute-style saving.
%   Supports:
%   - Numeric vectors (1D): split by element
%   - Numeric matrices (2D): split by row
%   - Cell arrays: split by element
%   - MATLAB tables: split by row (each row becomes a single-row table)
%   - String vectors (1D): split by element
%
%   Returns a cell array of individual pieces.
    if istable(data)
        pieces = cell(1, height(data));
        for i = 1:height(data)
            pieces{i} = data(i, :);
        end
    elseif isnumeric(data) || islogical(data)
        if isvector(data)
            pieces = cell(1, numel(data));
            for i = 1:numel(data)
                pieces{i} = data(i);
            end
        elseif ismatrix(data)
            pieces = cell(1, size(data, 1));
            for i = 1:size(data, 1)
                pieces{i} = data(i, :);
            end
        else
            error('scidb:for_each', ...
                'distribute does not support arrays with %d dimensions. Only vectors and matrices are supported.', ...
                ndims(data));
        end
    elseif iscell(data)
        pieces = data(:)';  % ensure row cell array
    elseif isstring(data)
        pieces = cellstr(data)';
    else
        error('scidb:for_each', ...
            'distribute does not support type %s. Supported types: numeric array, cell array, table.', ...
            class(data));
    end
end


function tbl = fe_multi_result_to_table(results, type_name)
%FE_MULTI_RESULT_TO_TABLE  Convert an array of ThunkOutput to a MATLAB table.
    n = numel(results);

    % Sort results by metadata for deterministic row order
    if n > 1
        meta_fields = sort(fieldnames(results(1).metadata));
        keys = strings(n, 1);
        for i = 1:n
            parts = cell(1, numel(meta_fields));
            for f = 1:numel(meta_fields)
                val = results(i).metadata.(meta_fields{f});
                if isnumeric(val)
                    parts{f} = sprintf('%s=%020.10f', meta_fields{f}, val);
                else
                    parts{f} = sprintf('%s=%s', meta_fields{f}, string(val));
                end
            end
            keys(i) = strjoin(string(parts), '|');
        end
        [~, order] = sort(keys);
        results = results(order);
    end

    % Check if all data items are tables — if so, flatten into top-level columns
    all_tables = true;
    for i = 1:n
        if ~istable(results(i).data)
            all_tables = false;
            break;
        end
    end

    if all_tables
        % Flatten mode: metadata columns + data columns merged into one table.
        % Each result may have multiple rows; metadata is replicated per row.
        rows = cell(n, 1);
        for i = 1:n
            data_tbl = results(i).data;
            nr = height(data_tbl);

            % Build metadata columns for this result (replicated per row)
            meta_tbl = table();
            meta_fields = fieldnames(results(i).metadata);
            for f = 1:numel(meta_fields)
                if isfield(results(i).metadata, meta_fields{f})
                    val = results(i).metadata.(meta_fields{f});
                else
                    val = missing;
                end
                if isnumeric(val)
                    meta_tbl.(meta_fields{f}) = repmat(double(val), nr, 1);
                elseif isstring(val) || ischar(val)
                    meta_tbl.(meta_fields{f}) = repmat(string(val), nr, 1);
                else
                    meta_tbl.(meta_fields{f}) = repmat({val}, nr, 1);
                end
            end

            % Concatenate metadata + data columns for this result
            rows{i} = [meta_tbl, data_tbl];
        end

        tbl = vertcat(rows{:});
    else
        % Non-table data: nest into a cell column named after the variable type
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
            tbl.(meta_fields{f}) = normalize_cell_column(col_data);
        end

        % Data column (named after the variable type)
        parts = strsplit(type_name, '.');
        col_name = parts{end};
        data_col = cell(n, 1);
        for i = 1:n
            data_col{i} = results(i).data;
        end
        tbl.(col_name) = normalize_cell_column(data_col);
    end
end


function result = apply_column_selection(loaded_val, cols, param_name)
%APPLY_COLUMN_SELECTION  Extract selected columns from a loaded variable.
%
%   For single column: returns the column contents as a vector/cell array.
%   For multiple columns: returns a MATLAB subtable.
%
%   Works on:
%   - scidb.ThunkOutput with .data = MATLAB table
%   - MATLAB table directly (from fe_multi_result_to_table)
%
%   Raises an error if the data is not a MATLAB table.

    % Get the table — either directly or from inside a ThunkOutput
    if isa(loaded_val, 'scidb.ThunkOutput')
        if numel(loaded_val) == 1
            tbl = loaded_val.data;
        else
            if istable(loaded_val(1).data)
                parts = cell(numel(loaded_val), 1);
                for i = 1:numel(loaded_val)
                    parts{i} = loaded_val(i).data;
                end
                tbl = vertcat(parts{:});
            else
                % Array of ThunkOutputs — not yet supported for column selection
                error('scidb:for_each', ...
                    'Column selection on ''%s'' is not supported for multi-result non-table arrays. Use as_table=true first.', ...
                    param_name);
            end
        end
    elseif istable(loaded_val)
        tbl = loaded_val;
    else
        error('scidb:for_each', ...
            'Column selection on ''%s'' requires table data, but loaded data is %s.', ...
            param_name, class(loaded_val));
    end

    if ~istable(tbl)
        error('scidb:for_each', ...
            'Column selection on ''%s'' requires table data, but loaded data is %s.', ...
            param_name, class(tbl));
    end

    % Validate column names
    for i = 1:numel(cols)
        if ~ismember(cols(i), tbl.Properties.VariableNames)
            error('scidb:for_each', ...
                'Column ''%s'' not found in ''%s''. Available columns: %s', ...
                cols(i), param_name, strjoin(tbl.Properties.VariableNames, ', '));
        end
    end

    if numel(cols) == 1
        % Single column: return raw array/cell
        col_data = tbl.(cols(1));
        if isscalar(loaded_val) && isa(loaded_val, 'scidb.ThunkOutput')
            loaded_val.data = col_data;
            result = loaded_val;
        else
            result = col_data;
        end
    else
        % Multiple columns: return subtable
        sub = tbl(:, cols);
        if isscalar(loaded_val) && isa(loaded_val, 'scidb.ThunkOutput')
            loaded_val.data = sub;
            result = loaded_val;
        else
            result = sub;
        end
    end
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


function result = merge_constituents(merge_spec, meta_nv, db_nv, param_name)
%MERGE_CONSTITUENTS  Load and merge Merge constituents into a MATLAB table.
%
%   For each constituent in the Merge:
%   - Resolve Fixed wrappers and column selection
%   - Load the variable (per-iteration, no preload for Merge)
%   - Build a keyed table with schema key columns + data columns
%   - Inner-join all constituent tables on common schema keys
%
%   When constituents return multiple records, they are joined by their
%   common schema keys (inner join). Unmatched rows are dropped.

    n = numel(merge_spec.var_specs);
    all_loaded = cell(1, n);
    all_col_sel = cell(1, n);
    all_short_names = cell(1, n);
    all_fixed_keys = cell(1, n);

    for i = 1:n
        spec = merge_spec.var_specs{i};

        [var_inst, load_nv, col_sel] = resolve_merge_spec(spec, meta_nv);

        type_name = class(var_inst);
        type_parts = strsplit(type_name, '.');
        all_short_names{i} = type_parts{end};
        all_col_sel{i} = col_sel;

        % Track which schema keys are fixed for this constituent
        if isa(spec, 'scidb.Fixed')
            all_fixed_keys{i} = string(fieldnames(spec.fixed_metadata));
        else
            all_fixed_keys{i} = string.empty;
        end

        % Load the variable
        all_loaded{i} = var_inst.load(load_nv{:}, db_nv{:});
    end

    result = merge_by_schema_keys(all_loaded, all_col_sel, all_short_names, all_fixed_keys, db_nv, param_name);
end


function tbl = to_table_part(raw, var_name, param_name, idx)
%TO_TABLE_PART  Convert raw data to a table fragment for merging.
    if istable(raw)
        tbl = raw;
    elseif isnumeric(raw) && isscalar(raw)
        tbl = table(raw, 'VariableNames', {var_name});
    elseif isnumeric(raw) && isvector(raw)
        tbl = table(raw(:), 'VariableNames', {var_name});
    elseif isnumeric(raw) && ismatrix(raw)
        % 2D matrix: one column per matrix column
        tbl = table();
        for j = 1:size(raw, 2)
            col_name = sprintf('%s_%d', var_name, j-1);
            tbl.(col_name) = raw(:, j);
        end
    elseif isstring(raw) || ischar(raw)
        if ischar(raw)
            raw = string(raw);
        end
        if isscalar(raw)
            tbl = table(raw, 'VariableNames', {var_name});
        else
            tbl = table(raw(:), 'VariableNames', {var_name});
        end
    elseif iscell(raw)
        tbl = table(raw(:), 'VariableNames', {var_name});
    elseif islogical(raw)
        if isscalar(raw)
            tbl = table(raw, 'VariableNames', {var_name});
        else
            tbl = table(raw(:), 'VariableNames', {var_name});
        end
    else
        error('scidb:Merge', ...
            'Merge constituent %s[%d] has unsupported data type %s. Supported: table, numeric, string, cell, logical.', ...
            param_name, idx, class(raw));
    end
end


function [var_inst, load_nv, col_sel] = resolve_merge_spec(spec, meta_nv)
%RESOLVE_MERGE_SPEC  Extract variable instance, load metadata, and column
%   selection from a Merge constituent spec.
    col_sel = string.empty;
    if isa(spec, 'scidb.Fixed')
        inner = spec.var_type;
        load_nv = meta_nv;
        ff = fieldnames(spec.fixed_metadata);
        for f = 1:numel(ff)
            load_nv{end+1} = ff{f}; %#ok<AGROW>
            load_nv{end+1} = spec.fixed_metadata.(ff{f}); %#ok<AGROW>
        end
        if isa(inner, 'scidb.BaseVariable') && ~isempty(inner.selected_columns)
            col_sel = inner.selected_columns;
        end
        var_inst = inner;
    elseif isa(spec, 'scidb.BaseVariable') && ~isempty(spec.selected_columns)
        col_sel = spec.selected_columns;
        var_inst = spec;
        load_nv = meta_nv;
    else
        var_inst = spec;
        load_nv = meta_nv;
    end
end


function raw = apply_merge_col_sel(raw, col_sel, short_name)
%APPLY_MERGE_COL_SEL  Apply column selection to raw data for Merge.
    if istable(raw)
        for ci = 1:numel(col_sel)
            if ~ismember(col_sel(ci), raw.Properties.VariableNames)
                error('scidb:Merge', ...
                    'Column ''%s'' not found in ''%s''. Available: %s', ...
                    col_sel(ci), short_name, strjoin(raw.Properties.VariableNames, ', '));
            end
        end
        if numel(col_sel) == 1
            col_data = raw.(col_sel(1));
            raw = table(col_data, 'VariableNames', {char(col_sel(1))});
        else
            raw = raw(:, col_sel);
        end
    else
        error('scidb:Merge', ...
            'Column selection on ''%s'' requires table data, but loaded data is %s.', ...
            short_name, class(raw));
    end
end


function result = merge_by_schema_keys(all_loaded, all_col_sel, all_short_names, all_fixed_keys, db_nv, param_name)
%MERGE_BY_SCHEMA_KEYS  Match records by schema keys and merge column-wise.
%
%   Uses record-level matching (not row-level) to avoid cross-products
%   when a single record contains multi-row table data.
%
%   1. Determine common schema keys across all constituents
%      (excluding keys that are Fixed in any constituent)
%   2. Build per-constituent lookup: common_key_str → record indices
%   3. Inner join: find key combinations present in ALL constituents
%   4. For each matched key: cross-product matching records, merge data
%      column-wise, add schema key columns
%   5. Stack all results

    n = numel(all_loaded);

    % Get schema keys from the database
    if ~isempty(db_nv) && numel(db_nv) >= 2
        py_db = db_nv{2};
    else
        py_db = py.scidb.database.get_database();
    end
    sk_cell = cell(py_db.dataset_schema_keys);
    schema_keys = string.empty;
    for s = 1:numel(sk_cell)
        schema_keys(end+1) = string(sk_cell{s}); %#ok<AGROW>
    end

    % Determine which schema keys each constituent has in its metadata
    per_const_keys = cell(1, n);
    for i = 1:n
        results = all_loaded{i};
        meta_fields = string(fieldnames(results(1).metadata));
        per_const_keys{i} = intersect(schema_keys, meta_fields);
    end

    % Keys that are fixed in ANY constituent — exclude from join
    any_fixed = string.empty;
    for i = 1:n
        any_fixed = union(any_fixed, all_fixed_keys{i});
    end

    % Common schema keys (intersection across all constituents, minus fixed)
    common_keys = sort(per_const_keys{1});
    for i = 2:n
        common_keys = intersect(common_keys, per_const_keys{i});
    end
    common_keys = setdiff(common_keys, any_fixed);

    % Check for data column conflicts upfront (using first record of each)
    all_data_cols = {};
    for i = 1:n
        raw = all_loaded{i}(1).data;
        if ~isempty(all_col_sel{i})
            raw = apply_merge_col_sel(raw, all_col_sel{i}, all_short_names{i});
        end
        sample = to_table_part(raw, all_short_names{i}, param_name, i-1);
        cols = setdiff(string(sample.Properties.VariableNames), schema_keys);
        for j = 1:numel(cols)
            if ismember(char(cols(j)), all_data_cols)
                error('scidb:Merge', ...
                    'Column name conflict in Merge for ''%s'': column ''%s'' appears in multiple constituents. Use column selection to select non-conflicting columns.', ...
                    param_name, cols(j));
            end
            all_data_cols{end+1} = char(cols(j)); %#ok<AGROW>
        end
    end

    % Build per-constituent lookup: common_key_str → array of record indices
    maps = cell(1, n);
    for i = 1:n
        map = containers.Map('KeyType', 'char', 'ValueType', 'any');
        for j = 1:numel(all_loaded{i})
            key = schema_key_str(all_loaded{i}(j).metadata, common_keys);
            if map.isKey(key)
                map(key) = [map(key), j];
            else
                map(key) = j;
            end
        end
        maps{i} = map;
    end

    % Inner join: find keys present in ALL constituents
    matched_keys = string(maps{1}.keys);
    for i = 2:n
        matched_keys = intersect(matched_keys, string(maps{i}.keys));
    end

    if isempty(matched_keys)
        error('scidb:Merge', ...
            'No matching records found across Merge constituents for ''%s''.', ...
            param_name);
    end
    matched_keys = sort(matched_keys);

    % For each matched key, cross-product matching records, merge data
    all_rows = cell(numel(matched_keys), 1);
    for k = 1:numel(matched_keys)
        key = char(matched_keys(k));

        % Get matching record indices from each constituent
        idx_per_const = cell(1, n);
        for i = 1:n
            idx_per_const{i} = maps{i}(key);
        end

        % Cross product of record indices
        combos = cartesian_indices(idx_per_const);

        combo_rows = cell(size(combos, 1), 1);
        for c = 1:size(combos, 1)
            parts = cell(1, n);
            all_meta = struct();

            for i = 1:n
                rec = all_loaded{i}(combos(c, i));
                raw = rec.data;

                % Apply column selection
                if ~isempty(all_col_sel{i})
                    raw = apply_merge_col_sel(raw, all_col_sel{i}, all_short_names{i});
                end

                parts{i} = to_table_part(raw, all_short_names{i}, param_name, i-1);

                % Collect schema key values from this record, but skip
                % fixed keys from Fixed constituents (prefer non-fixed
                % constituent values for those keys)
                meta_fields = fieldnames(rec.metadata);
                for mf = 1:numel(meta_fields)
                    if ismember(string(meta_fields{mf}), schema_keys)
                        is_fixed_here = ismember(string(meta_fields{mf}), all_fixed_keys{i});
                        already_set = isfield(all_meta, meta_fields{mf});
                        if ~is_fixed_here || ~already_set
                            all_meta.(meta_fields{mf}) = rec.metadata.(meta_fields{mf});
                        end
                    end
                end
            end

            % Column-wise merge with broadcast
            merged = merge_parts_columnwise(parts, all_short_names, param_name);
            nr = height(merged);

            % Build schema key columns (preserving schema order)
            meta_tbl = table();
            sk_present = intersect(schema_keys, string(fieldnames(all_meta)), 'stable');
            for sk = 1:numel(sk_present)
                sk_name = char(sk_present(sk));
                val = all_meta.(sk_name);
                if isnumeric(val)
                    meta_tbl.(sk_name) = repmat(double(val), nr, 1);
                else
                    meta_tbl.(sk_name) = repmat(string(val), nr, 1);
                end
            end

            % Remove data columns that overlap with schema key columns
            overlap = intersect( ...
                string(merged.Properties.VariableNames), ...
                string(meta_tbl.Properties.VariableNames));
            if ~isempty(overlap)
                merged(:, cellstr(overlap)) = [];
            end

            combo_rows{c} = [meta_tbl, merged];
        end

        all_rows{k} = vertcat(combo_rows{:});
    end

    result = vertcat(all_rows{:});
end


function key = schema_key_str(metadata, keys)
%SCHEMA_KEY_STR  Build a sorted lookup key from metadata schema key values.
    parts = cell(1, numel(keys));
    for k = 1:numel(keys)
        kname = char(keys(k));
        if isfield(metadata, kname)
            val = metadata.(kname);
            if isnumeric(val)
                parts{k} = sprintf('%s=%g', kname, val);
            else
                parts{k} = sprintf('%s=%s', kname, string(val));
            end
        else
            parts{k} = sprintf('%s=<missing>', kname);
        end
    end
    key = char(strjoin(string(parts), '|'));
end


function result = merge_parts_columnwise(parts, part_names, param_name)
%MERGE_PARTS_COLUMNWISE  Merge table fragments column-wise with broadcast.
%   Validates no column conflicts and consistent row counts.
%   Broadcasts single-row tables to match multi-row tables.

    % Check column name conflicts
    seen = {};
    for i = 1:numel(parts)
        cols = parts{i}.Properties.VariableNames;
        for c = 1:numel(cols)
            if ismember(cols{c}, seen)
                error('scidb:Merge', ...
                    'Column name conflict in Merge for ''%s'': column ''%s'' appears in multiple constituents. Use column selection to select non-conflicting columns.', ...
                    param_name, cols{c});
            end
            seen{end+1} = cols{c}; %#ok<AGROW>
        end
    end

    % Determine target row count from multi-row parts
    row_counts = cellfun(@height, parts);
    multi_row = row_counts(row_counts > 1);

    if ~isempty(multi_row)
        unique_counts = unique(multi_row);
        if numel(unique_counts) > 1
            detail_parts = cell(1, numel(parts));
            for i = 1:numel(parts)
                detail_parts{i} = sprintf('%s=%d', part_names{i}, row_counts(i));
            end
            error('scidb:Merge', ...
                'Cannot merge constituents with different row counts in ''%s'': %s. All multi-row constituents must have the same number of rows.', ...
                param_name, strjoin(detail_parts, ', '));
        end
        target_len = unique_counts(1);
    else
        target_len = 1;
    end

    % Broadcast single-row parts and concatenate
    result = table();
    for i = 1:numel(parts)
        tbl_i = parts{i};
        if height(tbl_i) == 1 && target_len > 1
            tbl_i = repmat(tbl_i, target_len, 1);
        end
        result = [result, tbl_i]; %#ok<AGROW>
    end
end


function col = normalize_cell_column(col_data)
%NORMALIZE_CELL_COLUMN  Convert a cell column to its native type.
%   All scalar numeric cells → numeric array, all scalar string/char cells →
%   string array, otherwise leave as cell array.
    n = numel(col_data);
    all_scalar_numeric = true;
    all_string = true;
    for i = 1:n
        v = col_data{i};
        if ~(isnumeric(v) && isscalar(v))
            all_scalar_numeric = false;
        end
        if ~(isstring(v) || ischar(v))
            all_string = false;
        end
    end
    if all_scalar_numeric
        col = cell2mat(col_data);
    elseif all_string
        col = string(col_data);
    else
        col = col_data;
    end
end


function combos = cartesian_indices(idx_cells)
%CARTESIAN_INDICES  Cartesian product of index arrays.
%   idx_cells: 1×n cell array, each containing an array of indices.
%   Returns: total×n matrix where each row is one combination.
    n_dims = numel(idx_cells);
    if n_dims == 1
        combos = idx_cells{1}(:);
        return;
    end

    sizes = cellfun(@numel, idx_cells);
    total = prod(sizes);
    combos = zeros(total, n_dims);

    repeats_after = 1;
    for d = n_dims:-1:1
        vals = idx_cells{d}(:);
        nv = numel(vals);
        repeats_before = total / (repeats_after * nv);
        col = repmat(repelem(vals, repeats_after), repeats_before, 1);
        combos(:, d) = col;
        repeats_after = repeats_after * nv;
    end
end
