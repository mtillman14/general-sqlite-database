function results = run_tests()
%RUN_TESTS  Run all MATLAB integration tests for sci-matlab.
%
%   results = run_tests()
%
%   Sets up paths, discovers all Test*.m files in scifor/, scidb/, and
%   scihist/ subdirectories, and runs them using MATLAB's unittest framework.
%
%   Example:
%       cd sci-matlab/tests/matlab
%       results = run_tests();

    % clear functions;
    % rehash path;
    % 
    % pythonPath = '/Users/mitchelltillman/Documents/test-project/.venv/bin/python3.11';
    % pyenv('Version', pythonPath);

    this_dir = fileparts(mfilename('fullpath'));
    addpath(genpath(this_dir));

    % Set up MATLAB and Python paths
    run(fullfile(this_dir, 'setup_paths.m'));

    % Discover and run all test classes (scifor + scidb + scihist layers)
    import matlab.unittest.TestSuite
    import matlab.unittest.TestRunner
    import matlab.unittest.plugins.DiagnosticsValidationPlugin

    suite_scifor  = TestSuite.fromFolder(fullfile(this_dir, 'scifor'));
    suite_scidb   = TestSuite.fromFolder(fullfile(this_dir, 'scidb'));
    suite_scihist = TestSuite.fromFolder(fullfile(this_dir, 'scihist'));
    suite = [suite_scifor, suite_scidb, suite_scihist];
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
