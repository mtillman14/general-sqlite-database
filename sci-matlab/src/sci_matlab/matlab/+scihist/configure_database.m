function db = configure_database(db_path, schema_keys, varargin)
%SCIHIST.CONFIGURE_DATABASE  Configure database and register it with Thunk.query.
%
%   DB = scihist.configure_database(DB_PATH, SCHEMA_KEYS, ...)
%
%   This is the scihist wrapper around scidb.configure_database(). It
%   opens the DuckDB-backed database AND sets py.thunk.Thunk.query = db
%   so that Thunk-based computations can look up previously computed
%   results (enabling cache hits).
%
%   Use this function (instead of scidb.configure_database) wherever
%   Thunk caching or provenance tracking is required.
%
%   Arguments:
%       db_path     - Path to the DuckDB database file (string or char)
%       schema_keys - String array of metadata keys that form the dataset
%                     schema (e.g. ["subject", "session"])
%       varargin    - Additional name-value pairs forwarded to
%                     scidb.configure_database (e.g. lineage_mode="strict")
%
%   Returns:
%       DB - The configured DatabaseManager Python object.
%
%   Example:
%       db = scihist.configure_database("experiment.duckdb", ...
%           ["subject", "session"]);

    db = py.scidb.configure_database(db_path, py.list(schema_keys), varargin{:});
    py.thunk.Thunk.query = db;
end
