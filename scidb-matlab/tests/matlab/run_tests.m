function results = run_tests()
%RUN_TESTS  Run all MATLAB integration tests for scidb-matlab.
%
%   results = run_tests()
%
%   Sets up paths, discovers all Test*.m files in this directory,
%   and runs them using MATLAB's unittest framework.
%
%   Example:
%       cd scidb-matlab/tests/matlab
%       results = run_tests();

    % clear functions;
    % rehash path;
    % 
    % pythonPath = '/Users/mitchelltillman/Documents/test-project/.venv/bin/python3.11';
    % pyenv('Version', pythonPath);

    this_dir = fileparts(mfilename('fullpath'));

    % Set up MATLAB and Python paths
    run(fullfile(this_dir, 'setup_paths.m'));

    % Discover and run all test classes
    import matlab.unittest.TestSuite
    import matlab.unittest.TestRunner
    import matlab.unittest.plugins.DiagnosticsValidationPlugin

    suite = TestSuite.fromFolder(this_dir, 'IncludingSubfolders', false);
    runner = TestRunner.withTextOutput('Verbosity', 3);

    results = runner.run(suite);

    % Summary
    fprintf('\n=== Test Summary ===\n');
    fprintf('Total:  %d\n', numel(results));
    fprintf('Passed: %d\n', sum([results.Passed]));
    fprintf('Failed: %d\n', sum([results.Failed]));
    fprintf('Errors: %d\n', sum(~[results.Passed] & ~[results.Failed]));

    if all([results.Passed])
        fprintf('\nAll tests passed.\n');
    else
        fprintf('\nFailing tests:\n');
        failed = results(~[results.Passed]);
        for i = 1:numel(failed)
            fprintf('  - %s\n', failed(i).Name);
        end
    end
end
