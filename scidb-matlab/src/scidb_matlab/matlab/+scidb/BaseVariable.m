classdef BaseVariable < dynamicprops
%SCIDB.BASEVARIABLE  Base class for all database-storable variable types.
%
%   Define variable types as empty subclasses:
%
%       classdef RawSignal < scidb.BaseVariable
%       end
%
%   Then use instance methods for all database operations:
%
%       RawSignal().save(data, subject=1, session="A")
%       var = RawSignal().load(subject=1, session="A")
%
%   The class name becomes the database table name automatically via
%   class(obj).  No additional properties or methods are needed.
%
%   Properties (populated after load):
%       data         - The loaded data (MATLAB native type)
%       record_id    - Unique record identifier (string)
%       metadata     - Struct of metadata key-value pairs
%       content_hash - Content hash of the data (string)
%       lineage_hash - Lineage hash, if computed by a thunk (string)
%       py_obj       - Python BaseVariable shadow (used internally)

    properties
        data                    % MATLAB data
        record_id    string     % Unique record ID
        metadata     struct     % Metadata key-value pairs
        content_hash string     % Content hash (16-char hex)
        lineage_hash string     % Lineage hash (64-char hex), empty if raw
        py_obj                  % Python BaseVariable shadow (internal)
    end

    methods
        function obj = BaseVariable()
        %BASEVARIABLE  Construct an empty BaseVariable.
            obj.metadata = struct();
        end        

        % -----------------------------------------------------------------
        % save
        % -----------------------------------------------------------------
        function record_id = save(obj, data, varargin)
        %SAVE  Save data to the database under this variable type.
        %
        %   RECORD_ID = TypeClass().save(DATA, Name, Value, ...)
        %
        %   DATA can be a numeric array, scalar, scidb.ThunkOutput (lineage
        %   is stored automatically), or scidb.BaseVariable (re-save).
        %
        %   Name-Value Arguments:
        %       db - Optional DatabaseManager to use instead of the global
        %            database (returned by scidb.configure_database).
        %       Any other name-value pairs are metadata (e.g. subject=1).
        %
        %   Example:
        %       RawSignal().save(randn(100,3), subject=1, session="A");
        %
        %       result = my_thunk(input_var, 2.5);
        %       Processed().save(result, subject=1, session="A");
        %
        %       % Save to a specific database
        %       RawSignal().save(data, db=db2, subject=1);

            type_name = class(obj);
            py_class = scidb.internal.ensure_registered(type_name);

            % Marshal data to Python
            if isa(data, 'scidb.ThunkOutput')
                py_data = data.py_obj;
            elseif isa(data, 'scidb.BaseVariable')
                py_data = data.py_obj;
            else
                py_data = scidb.internal.to_python(data);
            end

            % Extract db option from metadata args
            [metadata_nv, db_val] = extract_db(varargin);
            py_kwargs = scidb.internal.metadata_to_pykwargs(metadata_nv{:});

            if isempty(db_val)
                py_db = py.scidb.database.get_database();
            else
                py_db = db_val;
            end
            py_record_id = py_db.save_variable(py_class, py_data, pyargs(py_kwargs{:}));
            record_id = char(py_record_id);


        end

        % -----------------------------------------------------------------
        % save_from_table
        % -----------------------------------------------------------------
        function record_ids = save_from_table(obj, tbl, data_column, metadata_columns, varargin)
        %SAVE_FROM_TABLE  Bulk-save each row of a MATLAB table as a separate record.
        %
        %   RECORD_IDS = TypeClass().save_from_table(TBL, DATA_COL, META_COLS, ...)
        %
        %   Uses a batched code path (~100x faster than looping save()).
        %
        %   Arguments:
        %       TBL              - MATLAB table where each row is a record
        %       DATA_COL         - Name of the column containing data values
        %                          (string or char)
        %       META_COLS        - Column names to use as per-row metadata
        %                          (string array or cell array of char)
        %
        %   Name-Value Arguments:
        %       db - Optional DatabaseManager to use instead of the global
        %            database (returned by scidb.configure_database).
        %       Any other name-value pairs are common metadata applied to
        %       every row (e.g. experiment="exp1").
        %
        %   Returns:
        %       String array of record_ids, one per row.
        %
        %   Example:
        %       % Table with 10 rows (2 subjects x 5 trials)
        %       %   Subject  Trial  MyVar
        %       %   1        1      0.5
        %       %   1        2      0.6
        %       %   ...
        %
        %       ids = ScalarValue().save_from_table( ...
        %           results_tbl, "MyVar", ["Subject", "Trial"], ...
        %           experiment="exp1");

            type_name = class(obj);
            scidb.internal.ensure_registered(type_name);

            % Normalise inputs
            if isstring(data_column), data_column = char(data_column); end
            if isstring(metadata_columns)
                metadata_columns = cellstr(metadata_columns);
            end

            % Separate db option from common metadata
            [common_nv, db_val] = extract_db(varargin);

            % --- Convert data column to Python (numpy for numeric) ---
            data_col = tbl.(data_column);
            if isnumeric(data_col)
                py_data = py.numpy.array(data_col(:)');
            elseif isstring(data_col)
                py_data = py.list(cellfun(@char, num2cell(data_col(:)'), ...
                    'UniformOutput', false));
            elseif iscellstr(data_col) %#ok<ISCLSTR>
                py_data = py.list(data_col(:)');
            else
                % Generic fallback: convert each element individually
                py_data = py.list();
                for i = 1:height(tbl)
                    py_data.append(scidb.internal.to_python(data_col(i)));
                end
            end

            % --- Convert metadata columns (numpy for numeric) ---
            py_meta_keys = py.list(metadata_columns);
            py_meta_cols = py.list();
            for j = 1:numel(metadata_columns)
                col = tbl.(metadata_columns{j});
                if iscategorical(col)
                    col = string(col); % Can't convert categorical to Python
                end
                if isnumeric(col)
                    py_meta_cols.append(py.numpy.array(col(:)'));
                elseif isstring(col)
                    % Join into single string (1 boundary crossing vs N)
                    py_meta_cols.append(strjoin(col(:)', char(30)));
                elseif iscellstr(col) %#ok<ISCLSTR>
                    py_meta_cols.append(strjoin(string(col(:)'), char(30)));
                else
                    py_col = py.list();
                    for i = 1:height(tbl)
                        py_col.append(scidb.internal.to_python(col(i)));
                    end
                    py_meta_cols.append(py_col);
                end
            end

            % --- Build common metadata dict ---
            py_common = scidb.internal.metadata_to_pydict(common_nv{:});

            % --- Database ---
            if isempty(db_val)
                py_db = py.None;
            else
                py_db = db_val;
            end

            % --- Call Python bridge ---
            py_result = py.scidb_matlab.bridge.save_batch_bridge( ...
                type_name, py_data, py_meta_keys, py_meta_cols, ...
                py_common, py_db);

            % --- Convert result to MATLAB string array ---
            record_ids = splitlines(string(py_result));

        end

        % -----------------------------------------------------------------
        % load
        % -----------------------------------------------------------------
        function result = load(obj, varargin)
        %LOAD  Load a variable from the database.
        %
        %   RESULT = TypeClass().load(Name, Value, ...)
        %
        %   Returns a scidb.ThunkOutput with .data, .record_id, .metadata,
        %   matching the return type of scidb.Thunk calls.  The Python
        %   BaseVariable shadow is stored in .py_obj so that lineage
        %   tracking works when the result is passed to another thunk or
        %   re-saved.
        %
        %   Name-Value Arguments:
        %       Any metadata key-value pairs (e.g. subject=1, session="A")
        %       version  - Specific record_id to load (default "latest")
        %       as_table - If true, return a MATLAB table when multiple
        %                  results match (default false)
        %       db       - Optional DatabaseManager to use instead of the
        %                  global database
        %
        %   Example:
        %       raw = RawSignal().load(subject=1, session="A");
        %       disp(raw.data);
        %
        %       % Load as table
        %       tbl = RawSignal().load(as_table=true, subject=1);
        %
        %       % Load from a specific database
        %       raw = RawSignal().load(db=db2, subject=1, session="A");

            type_name = class(obj);
            py_class = scidb.internal.ensure_registered(type_name);

            [metadata_args, version, as_table, db_val] = split_load_args(varargin{:});
            py_metadata = scidb.internal.metadata_to_pydict(metadata_args{:});

            if isempty(db_val)
                py_db = py.scidb.database.get_database();
            else
                py_db = db_val;
            end

            % If loading by specific version, always return single
            if version ~= "latest"
                py_var = py_db.load(py_class, py_metadata, version=char(version));
                result = scidb.BaseVariable.wrap_py_var(py_var);
                return;
            end

            % Query all matching records (latest version per parameter set)
            py_gen = py_db.load_all(py_class, py_metadata, pyargs('version_id', 'latest'));
            py_list = py.list(py_gen);
            n = int64(py.builtins.len(py_list));

            if n == 0
                error('scidb:NotFoundError', 'No %s found matching the given metadata.', type_name);
            elseif n == 1
                % Single match → return single ThunkOutput
                result = scidb.BaseVariable.wrap_py_var(py_list{1});
            else
                % Multiple matches
                results_arr = scidb.ThunkOutput.empty();
                for i = 1:n
                    results_arr(end+1) = scidb.BaseVariable.wrap_py_var(py_list{i}); %#ok<AGROW>
                end

                if as_table
                    result = multi_result_to_table(results_arr, type_name);
                else
                    result = results_arr;
                end
            end


        end

        % -----------------------------------------------------------------
        % load_all
        % -----------------------------------------------------------------
        function results = load_all(obj, varargin)
        %LOAD_ALL  Load all variables matching the given metadata.
        %
        %   RESULTS = TypeClass().load_all(Name, Value, ...)
        %
        %   Returns an array of scidb.ThunkOutput objects.
        %
        %   Name-Value Arguments:
        %       version_id - Which versions to return (default "all"):
        %           "all"    : return every version
        %           "latest" : return only the latest version per parameter set
        %           integer  : return only that specific version_id
        %       db         - Optional DatabaseManager to use instead of the
        %                    global database
        %       Any other name-value pairs are metadata filters.
        %       Non-scalar numeric or string arrays are treated as "match any".
        %
        %   Example:
        %       all_signals = RawSignal().load_all(subject=1);
        %       latest_only = RawSignal().load_all(subject=1, version_id="latest");
        %       all_from_db = RawSignal().load_all(db=db2, subject=1);

            type_name = class(obj);
            py_class = scidb.internal.ensure_registered(type_name);

            [metadata_args, py_version_id, as_table, db_val] = scidb.internal.split_load_all_args(varargin{:});
            py_metadata = scidb.internal.metadata_to_pydict(metadata_args{:});

            if isempty(db_val)
                py_db = py.scidb.database.get_database();
            else
                py_db = db_val;
            end
            py_gen = py_db.load_all(py_class, py_metadata, pyargs('version_id', py_version_id));

            results_arr = scidb.ThunkOutput.empty();
            py_iter = py.builtins.iter(py_gen);

            while true
                try
                    py_var = py.builtins.next(py_iter);
                catch
                    break;
                end
                results_arr(end+1) = scidb.BaseVariable.wrap_py_var(py_var); %#ok<AGROW>
            end

            if as_table && numel(results_arr) > 1
                results = multi_result_to_table(results_arr, type_name);
            else
                results = results_arr;
            end

        end

        % -----------------------------------------------------------------
        % list_versions
        % -----------------------------------------------------------------
        function versions = list_versions(obj, varargin)
        %LIST_VERSIONS  List all versions at a schema location.
        %
        %   VERSIONS = TypeClass().list_versions(Name, Value, ...)
        %
        %   Name-Value Arguments:
        %       db - Optional DatabaseManager to use instead of the global
        %            database
        %       Any other name-value pairs are metadata filters.
        %
        %   Returns a struct array with fields: record_id, schema,
        %   version, created_at.
        %
        %   Example:
        %       v = ProcessedSignal().list_versions(subject=1, session="A");

            type_name = class(obj);
            py_class = scidb.internal.ensure_registered(type_name);

            [metadata_nv, db_val] = extract_db(varargin);
            py_kwargs = scidb.internal.metadata_to_pykwargs(metadata_nv{:});

            if isempty(db_val)
                py_db = py.scidb.database.get_database();
            else
                py_db = db_val;
            end
            py_list = py_db.list_versions(py_class, pyargs(py_kwargs{:}));

            n = int64(py.builtins.len(py_list));
            versions = struct('record_id', {}, 'schema', {}, 'version', {}, 'created_at', {});

            for i = 1:n
                py_dict = py_list{i};
                versions(i).record_id  = string(py_dict{'record_id'});
                versions(i).schema     = scidb.internal.pydict_to_struct(py_dict{'schema'});
                versions(i).version    = scidb.internal.pydict_to_struct(py_dict{'version'});
                versions(i).created_at = string(py_dict{'created_at'});
            end

            
        end

        % -----------------------------------------------------------------
        % provenance
        % -----------------------------------------------------------------
        function prov = provenance(obj, varargin)
        %PROVENANCE  Get the provenance (lineage) of a variable.
        %
        %   PROV = TypeClass().provenance(Name, Value, ...)
        %
        %   Name-Value Arguments:
        %       db - Optional DatabaseManager to use instead of the global
        %            database
        %       Any other name-value pairs are metadata filters.
        %
        %   Returns a struct with function_name, function_hash, inputs,
        %   constants.  Returns [] if no lineage recorded.
        %
        %   Example:
        %       p = ProcessedSignal().provenance(subject=1, session="A");

            type_name = class(obj);
            py_class = scidb.internal.ensure_registered(type_name);

            [metadata_args, version, db_val] = scidb.internal.split_version_arg(varargin{:});
            py_kwargs = scidb.internal.metadata_to_pykwargs(metadata_args{:});

            if isempty(db_val)
                py_db = py.scidb.database.get_database();
            else
                py_db = db_val;
            end
            if version ~= "latest"
                disp('what to do with syntax py_kwargs{:}?')
                % py_result = py_db.get_provenance(py_class, version=char(version), pyargs(py_kwargs{:}));
            else
                py_result = py_db.get_provenance(py_class, pyargs(py_kwargs{:}));
            end

            if isa(py_result, 'py.NoneType')
                prov = [];
            else
                prov.function_name = string(py_result{'function_name'});
                prov.function_hash = string(py_result{'function_hash'});
                prov.inputs        = scidb.internal.pylist_to_cell(py_result{'inputs'});
                prov.constants     = scidb.internal.pylist_to_cell(py_result{'constants'});
            end

            
        end

        % -----------------------------------------------------------------
        % disp
        % -----------------------------------------------------------------
        function disp(obj)
        %DISP  Display the BaseVariable.
            if isempty(obj.data)
                fprintf('  %s (empty)\n', class(obj));
            else
                fprintf('  %s [%s]\n', class(obj), obj.record_id);
                fprintf('    data: %s\n', class(obj.data));
                if ~isempty(fieldnames(obj.metadata))
                    fprintf('    metadata: ');
                    disp(obj.metadata);
                end
            end
        end
    end

    methods (Static)
        function objs = empty()
        %EMPTY  Create an empty BaseVariable array (for preallocation).
            objs = scidb.BaseVariable.empty(0, 0);
        end

        function obj = str2var(s)
            %STR2VAR  Convert a string to a BaseVariable (for flexibility).
            %   This allows users to write code like:
            %      RawSignal = BaseVariable.str2var("RawSignal");
            if isstring(s), s = char(s); end
            var_folder = fullfile(pwd, 'src', 'vars');
            classFileName = fullfile(var_folder, [s '.m']);            
            if ~isfile(classFileName)
                classDefText = sprintf('classdef %s < scidb.BaseVariable\nend\n', s);
                % Write the text to the file
                fid = fopen(classFileName, 'w');
                if fid == -1
                    error('Could not create class definition file: %s', classFileName);
                end
                fprintf(fid, '%s', classDefText);
                fclose(fid);
                % Refresh the MATLAB path to recognize the new class
                rehash;
            end

            py_class = scidb.internal.ensure_registered(s);            

            obj = eval([s '()']); % Call the newly created class
            obj.py_obj = py_class([]);  % Create an empty Python BaseVariable shadow
        end

        function v = wrap_py_var(py_var)
        %WRAP_PY_VAR  Convert a Python BaseVariable to a MATLAB ThunkOutput.
        %   This is used internally to convert results from the database into
        %   MATLAB objects.  The returned ThunkOutput has the .py_obj property set to
        %   the original Python BaseVariable shadow, so that lineage tracking works if it's passed to another thunk or re-saved.
        % Usage: v = scidb.BaseVariable.wrap_py_var(py_var)
            matlab_data = scidb.internal.from_python(py_var.data);
            v = scidb.ThunkOutput(matlab_data, py_var);
            v.record_id = string(py_var.record_id);
            v.content_hash = string(py_var.content_hash);

            py_lh = py_var.lineage_hash;
            if ~isa(py_lh, 'py.NoneType')
                v.lineage_hash = string(py_lh);
            end

            py_meta = py_var.metadata;
            if ~isa(py_meta, 'py.NoneType')
                v.metadata = scidb.internal.pydict_to_struct(py_meta);
            end

            py_vid = py.builtins.getattr(py_var, 'version_id', py.None);
            if ~isa(py_vid, 'py.NoneType')
                v.version_id = int64(py_vid);
            end

            py_pid = py.builtins.getattr(py_var, 'parameter_id', py.None);
            if ~isa(py_pid, 'py.NoneType')
                v.parameter_id = int64(py_pid);
            end
        end
    end
end


% =========================================================================
% Local helper functions
% =========================================================================

function [metadata_args, version, as_table, db] = split_load_args(varargin)
%SPLIT_LOAD_ARGS  Separate 'version', 'as_table', and 'db' from metadata args.
    version = "latest";
    as_table = false;
    db = [];
    metadata_args = {};

    i = 1;
    while i <= numel(varargin)
        key = varargin{i};
        if isstring(key), key = char(key); end

        if strcmpi(key, 'version') && i < numel(varargin)
            version = string(varargin{i+1});
            i = i + 2;
        elseif strcmpi(key, 'as_table') && i < numel(varargin)
            as_table = logical(varargin{i+1});
            i = i + 2;
        elseif strcmpi(key, 'db') && i < numel(varargin)
            db = varargin{i+1};
            i = i + 2;
        else
            metadata_args{end+1} = varargin{i};   %#ok<AGROW>
            metadata_args{end+1} = varargin{i+1};  %#ok<AGROW>
            i = i + 2;
        end
    end
end


function [remaining, db] = extract_db(args)
%EXTRACT_DB  Extract 'db' option from name-value pairs.
%   Returns remaining name-value pairs and the db value ([] if not found).
    db = [];
    remaining = {};

    i = 1;
    while i <= numel(args)
        key = args{i};
        if (ischar(key) || isstring(key)) && strcmpi(string(key), 'db') && i < numel(args)
            db = args{i+1};
            i = i + 2;
        else
            remaining{end+1} = args{i};   %#ok<AGROW>
            if i < numel(args)
                remaining{end+1} = args{i+1}; %#ok<AGROW>
                i = i + 2;
            else
                i = i + 1;
            end
        end
    end
end


function tbl = multi_result_to_table(results, type_name)
%MULTI_RESULT_TO_TABLE  Convert an array of ThunkOutput to a MATLAB table.
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
        % Try to convert to numeric if all values are numeric
        all_numeric = true;
        for i = 1:n
            if ~isnumeric(col_data{i}) || ismissing(col_data{i})
                all_numeric = false;
                break;
            end
        end
        if all_numeric
            tbl.(meta_fields{f}) = cell2mat(col_data);
        else
            tbl.(meta_fields{f}) = col_data;
        end
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
    % Strip package prefix (e.g. "mypackage.StepLength" → "StepLength")
    parts = strsplit(type_name, '.');
    col_name = parts{end};
    data_col = cell(n, 1);
    for i = 1:n
        data_col{i} = results(i).data;
    end
    tbl.(col_name) = data_col;
end
