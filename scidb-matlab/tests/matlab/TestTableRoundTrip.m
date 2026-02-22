classdef TestTableRoundTrip < matlab.unittest.TestCase
%TESTTABLEROUNDTRIP  Round-trip tests for table (DataFrame) variables.
%
%   Verifies that MATLAB tables with various column types survive a full
%   save → load cycle with correct DuckDB column types and values:
%
%     * Multi-row, scalar numeric    → DOUBLE[]   round-trip
%     * Multi-row, scalar string     → VARCHAR[]  round-trip
%     * 1-row, scalar numeric        → DOUBLE     round-trip
%     * 1-row, scalar string         → VARCHAR    round-trip
%     * Multi-row, matrix column     → DOUBLE[][] round-trip
%
%   Also verifies for_each(distribute=false) behaves identically to .save()
%   and for_each(distribute=true) saves each row as a separate record with
%   scalar (DOUBLE / VARCHAR) columns.

    properties
        test_dir
    end

    methods (TestClassSetup)
        function addPaths(~)
            this_dir = fileparts(mfilename('fullpath'));
            run(fullfile(this_dir, 'setup_paths.m'));
        end
    end

    methods (TestMethodSetup)
        function setupDatabase(testCase)
            testCase.test_dir = tempname;
            mkdir(testCase.test_dir);
            scidb.configure_database( ...
                fullfile(testCase.test_dir, 'test.duckdb'), ...
                ["subject", "session"]);
        end
    end

    methods (TestMethodTeardown)
        function cleanup(testCase)
            try
                scidb.get_database().close();
            catch
            end
            if isfolder(testCase.test_dir)
                rmdir(testCase.test_dir, 's');
            end
        end
    end

    methods (Test)

        % =================================================================
        % Direct .save() / .load() round-trip tests
        % =================================================================

        function test_save_multirow_scalar_double(testCase)
            %% Multi-row table, scalar double columns → DOUBLE[] round-trip.
            t = table([1.0; 2.0; 3.0], [4.0; 5.0; 6.0], ...
                'VariableNames', {'A', 'B'});
            TableVar().save(t, 'subject', 1, 'session', 'A');

            types = testCase.getColumnTypes('TableVar_data');
            testCase.verifyEqual(types('A'), "DOUBLE");
            testCase.verifyEqual(types('B'), "DOUBLE");

            result = TableVar().load('subject', 1, 'session', 'A');
            testCase.verifyTrue(istable(result.data));
            testCase.verifyEqual(height(result.data), 3);
            testCase.verifyEqual(result.data.A, [1;2;3], 'AbsTol', 1e-10);
            testCase.verifyEqual(result.data.B, [4;5;6], 'AbsTol', 1e-10);
        end

        function test_save_multirow_scalar_string(testCase)
            %% Multi-row table, string column → VARCHAR[] round-trip.
            t = table(["alice"; "bob"; "carol"], ...
                'VariableNames', {'name'});
            TableVar().save(t, 'subject', 2, 'session', 'A');

            types = testCase.getColumnTypes('TableVar_data');
            testCase.verifyEqual(types('name'), "VARCHAR");

            result = TableVar().load('subject', 2, 'session', 'A');
            testCase.verifyTrue(istable(result.data));
            testCase.verifyEqual(height(result.data), 3);
            testCase.verifyEqual(result.data.name, ["alice"; "bob"; "carol"]);
        end

        function test_save_onerow_scalar_double(testCase)
            %% 1-row table, scalar double → DOUBLE (not DOUBLE[]) round-trip.
            t = table(42.0, 'VariableNames', {'x'});
            TableVar().save(t, 'subject', 3, 'session', 'A');

            types = testCase.getColumnTypes('TableVar_data');
            testCase.verifyEqual(types('x'), "DOUBLE");

            result = TableVar().load('subject', 3, 'session', 'A');
            testCase.verifyTrue(istable(result.data));
            testCase.verifyEqual(height(result.data), 1);
            testCase.verifyEqual(result.data.x, 42.0, 'AbsTol', 1e-10);
        end

        function test_save_onerow_scalar_string(testCase)
            %% 1-row table, scalar string → VARCHAR (not VARCHAR[]) round-trip.
            t = table("hello", 'VariableNames', {'label'});
            TableVar().save(t, 'subject', 4, 'session', 'A');

            types = testCase.getColumnTypes('TableVar_data');
            testCase.verifyEqual(types('label'), "VARCHAR");

            result = TableVar().load('subject', 4, 'session', 'A');
            testCase.verifyTrue(istable(result.data));
            testCase.verifyEqual(height(result.data), 1);
            testCase.verifyEqual(result.data.label, "hello");
        end

        function test_save_multirow_matrix_column(testCase)
            %% Multi-row table, NxM matrix column → DOUBLE[][] round-trip.
            %  Each row of the table contains a 1×3 row vector; stored as
            %  DOUBLE[][] and loaded back as a 3×3 matrix column.
            t = table([1,2,3; 4,5,6; 7,8,9], 'VariableNames', {'vec'});
            TableVar().save(t, 'subject', 5, 'session', 'A');

            types = testCase.getColumnTypes('TableVar_data');
            testCase.verifyEqual(types('vec'), "DOUBLE[]");

            result = TableVar().load('subject', 5, 'session', 'A');
            testCase.verifyTrue(istable(result.data));
            testCase.verifyEqual(height(result.data), 3);
            testCase.verifyEqual(result.data.vec, [1,2,3; 4,5,6; 7,8,9], 'AbsTol', 1e-10);
        end

        function test_save_multirow_mixed_columns(testCase)
            %% Multi-row table with both double and string columns round-trips.
            t = table([10.0; 20.0; 30.0], ["x"; "y"; "z"], ...
                'VariableNames', {'val', 'tag'});
            TableVar().save(t, 'subject', 6, 'session', 'A');

            types = testCase.getColumnTypes('TableVar_data');
            testCase.verifyEqual(types('val'), "DOUBLE");
            testCase.verifyEqual(types('tag'), "VARCHAR");

            result = TableVar().load('subject', 6, 'session', 'A');
            testCase.verifyTrue(istable(result.data));
            testCase.verifyEqual(height(result.data), 3);
            testCase.verifyEqual(result.data.val, [10;20;30], 'AbsTol', 1e-10);
            testCase.verifyEqual(result.data.tag, ["x"; "y"; "z"]);
        end

        % =================================================================
        % for_each(distribute=false) — whole table saved as one DuckDB row
        % =================================================================

        function test_foreach_false_multirow_scalar(testCase)
            %% for_each(distribute=false) behaves identically to .save():
            %  a 3-row table is stored as 3 DuckDB rows with DOUBLE/VARCHAR columns.
            t = table([10.0; 20.0; 30.0], ["x"; "y"; "z"], ...
                'VariableNames', {'val', 'tag'});

            scidb.for_each(@noop_func, ...
                struct('x', t), ...
                {TableVar()}, ...
                'subject', 10, 'session', 'B', ...
                distribute=false);

            types = testCase.getColumnTypes('TableVar_data');
            testCase.verifyEqual(types('val'), "DOUBLE");
            testCase.verifyEqual(types('tag'), "VARCHAR");

            result = TableVar().load('subject', 10, 'session', 'B');
            testCase.verifyTrue(istable(result.data));
            testCase.verifyEqual(height(result.data), 3);
            testCase.verifyEqual(result.data.val, [10;20;30], 'AbsTol', 1e-10);
            testCase.verifyEqual(result.data.tag, ["x"; "y"; "z"]);
        end

        % =================================================================
        % for_each(distribute=true) — each row saved as a separate record
        % =================================================================

        function test_foreach_true_scalar_double_rows(testCase)
            %% for_each(distribute=true) with scalar numeric columns:
            %  each row of the output table is saved as a separate session
            %  record, giving DOUBLE columns per record (1-row table).
            t = table([1.0; 2.0; 3.0], [4.0; 5.0; 6.0], ...
                'VariableNames', {'A', 'B'});

            scidb.for_each(@noop_func, ...
                struct('x', t), ...
                {TableVar()}, ...
                'subject', 20, ...
                distribute=true);

            % Row k → session k; each record is a 1-row table.
            for sess = 1:3
                result = TableVar().load('subject', 20, 'session', sess);
                testCase.verifyTrue(istable(result.data));
                testCase.verifyEqual(height(result.data), 1);
                testCase.verifyEqual(result.data.A, t.A(sess), 'AbsTol', 1e-10);
                testCase.verifyEqual(result.data.B, t.B(sess), 'AbsTol', 1e-10);
            end
        end

        function test_foreach_true_scalar_string_rows(testCase)
            %% for_each(distribute=true) with a string column:
            %  each row goes to a separate session record with VARCHAR column.
            t = table(["alice"; "bob"; "carol"], ...
                'VariableNames', {'name'});

            scidb.for_each(@noop_func, ...
                struct('x', t), ...
                {TableVar()}, ...
                'subject', 21, ...
                distribute=true);

            expected = ["alice", "bob", "carol"];
            for sess = 1:3
                result = TableVar().load('subject', 21, 'session', sess);
                testCase.verifyTrue(istable(result.data));
                testCase.verifyEqual(height(result.data), 1);
                testCase.verifyEqual(result.data.name, expected(sess));
            end
        end

        function test_foreach_true_idempotent(testCase)
            %% Running for_each(distribute=true) twice with identical data
            %  must not create duplicate records.
            t = table([1.0; 2.0; 3.0], 'VariableNames', {'val'});

            for run = 1:2
                scidb.for_each(@noop_func, ...
                    struct('x', t), ...
                    {TableVar()}, ...
                    'subject', 22, ...
                    distribute=true);
            end

            all_results = TableVar().load_all();
            testCase.verifyEqual(numel(all_results), 3);

            for sess = 1:3
                result = TableVar().load('subject', 22, 'session', sess);
                testCase.verifyEqual(result.data.val, t.val(sess), 'AbsTol', 1e-10);
            end
        end

    end

    methods (Access = private)
        function types = getColumnTypes(testCase, table_name)
            %% Query DuckDB information_schema for column data types.
            db = scidb.get_database();
            duck = py.getattr(db, '_duck');
            rows = duck.fetchall( ...
                "SELECT column_name, data_type FROM information_schema.columns " + ...
                "WHERE table_name = ? ORDER BY ordinal_position", ...
                py.list({table_name}));

            types = containers.Map();
            rows_cell = cell(rows);
            for i = 1:numel(rows_cell)
                row = cell(rows_cell{i});
                col_name = string(row{1});
                col_type = string(row{2});
                if col_name ~= "record_id"
                    types(col_name) = col_type;
                end
            end
        end
    end
end
