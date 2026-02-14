classdef BaseVariable < handle
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
        %   Example:
        %       RawSignal().save(randn(100,3), subject=1, session="A");
        %
        %       result = my_thunk(input_var, 2.5);
        %       Processed().save(result, subject=1, session="A");

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

            py_kwargs = scidb.internal.metadata_to_pykwargs(varargin{:});

            py_db = py.scidb.database.get_database();            
            py_record_id = py_db.save_variable(py_class, py_data, pyargs(py_kwargs{:}));
            record_id = char(py_record_id);

            
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
        %       version - Specific record_id to load (default "latest")
        %
        %   Example:
        %       raw = RawSignal().load(subject=1, session="A");
        %       disp(raw.data);

            type_name = class(obj);
            py_class = scidb.internal.ensure_registered(type_name);

            [metadata_args, version] = scidb.internal.split_version_arg(varargin{:});
            py_metadata = scidb.internal.metadata_to_pydict(metadata_args{:});

            py_db = py.scidb.database.get_database();

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
                % Multiple matches → return array of ThunkOutput
                result = scidb.ThunkOutput.empty();
                for i = 1:n
                    result(end+1) = scidb.BaseVariable.wrap_py_var(py_list{i}); %#ok<AGROW>
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
        %       Any other name-value pairs are metadata filters.
        %       Non-scalar numeric or string arrays are treated as "match any".
        %
        %   Example:
        %       all_signals = RawSignal().load_all(subject=1);
        %       latest_only = RawSignal().load_all(subject=1, version_id="latest");

            type_name = class(obj);
            py_class = scidb.internal.ensure_registered(type_name);

            [metadata_args, py_version_id] = scidb.internal.split_load_all_args(varargin{:});
            py_metadata = scidb.internal.metadata_to_pydict(metadata_args{:});

            py_db = py.scidb.database.get_database();
            py_gen = py_db.load_all(py_class, py_metadata, pyargs('version_id', py_version_id));

            results = scidb.ThunkOutput.empty();
            py_iter = py.builtins.iter(py_gen);

            while true
                try
                    py_var = py.builtins.next(py_iter);
                catch
                    break;
                end
                results(end+1) = scidb.BaseVariable.wrap_py_var(py_var); %#ok<AGROW>
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
        %   Returns a struct array with fields: record_id, schema,
        %   version, created_at.
        %
        %   Example:
        %       v = ProcessedSignal().list_versions(subject=1, session="A");

            type_name = class(obj);
            py_class = scidb.internal.ensure_registered(type_name);
            py_kwargs = scidb.internal.metadata_to_pykwargs(varargin{:});

            py_db = py.scidb.database.get_database();
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
        %   Returns a struct with function_name, function_hash, inputs,
        %   constants.  Returns [] if no lineage recorded.
        %
        %   Example:
        %       p = ProcessedSignal().provenance(subject=1, session="A");

            type_name = class(obj);
            py_class = scidb.internal.ensure_registered(type_name);

            [metadata_args, version] = scidb.internal.split_version_arg(varargin{:});
            py_kwargs = scidb.internal.metadata_to_pykwargs(metadata_args{:});

            py_db = py.scidb.database.get_database();
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

        function v = wrap_py_var(py_var)
        %WRAP_PY_VAR  Convert a Python BaseVariable to a MATLAB ThunkOutput.
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
        end
    end
end
