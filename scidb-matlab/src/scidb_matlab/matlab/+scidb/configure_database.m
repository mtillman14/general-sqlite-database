function db = configure_database(dataset_db_path, dataset_schema_keys, pipeline_db_path, options)
%SCIDB.CONFIGURE_DATABASE  Set up the SciDB database connection.
%
%   db = scidb.configure_database(DB_PATH, SCHEMA_KEYS, PIPELINE_PATH)
%   configures the global database connection with DuckDB for data storage
%   and SQLite for lineage tracking.
%
%   db = scidb.configure_database(..., lineage_mode="ephemeral")
%   allows unsaved intermediate variables in lineage chains.
%
%   Arguments:
%       dataset_db_path     - Path to the DuckDB database file (string)
%       dataset_schema_keys - Metadata keys defining the dataset schema
%                             (string array, e.g. ["subject", "session"])
%       pipeline_db_path    - Path to the SQLite lineage database (string)
%
%   Name-Value Arguments:
%       lineage_mode - "strict" (default) or "ephemeral"
%
%   Example:
%       scidb.configure_database( ...
%           "experiment.duckdb", ...
%           ["subject", "session"], ...
%           "pipeline.db");

    arguments
        dataset_db_path     string
        dataset_schema_keys string
        pipeline_db_path    string
        options.lineage_mode string = "strict"
    end

    % Convert keys to row vector
    if size(dataset_schema_keys,1) > 1
        dataset_schema_keys = dataset_schema_keys';
    end

    % Convert MATLAB string array to Python list of strings
    py_schema_keys = py.list(cellstr(dataset_schema_keys));

    dataset_db_path = char(dataset_db_path);
    if ~scidb.isabsolute(dataset_db_path)
        dataset_db_path = fullfile(pwd, dataset_db_path);
    end

    pipeline_db_path = char(pipeline_db_path);
    if ~scidb.isabsolute(pipeline_db_path)
        pipeline_db_path = fullfile(pwd, pipeline_db_path);
    end

    % Call Python's configure_database
    db = py.scidb.configure_database( ...
        char(dataset_db_path), ...
        py_schema_keys, ...
        char(pipeline_db_path), ...
        pyargs('lineage_mode', char(options.lineage_mode)));        

    % Verify the Python environment is working
    py_db = py.scidb.database.get_database();
    if isempty(py_db)
        error('scidb:ConfigFailed', ...
            'configure_database() did not produce a valid DatabaseManager.');
    end
    
end
