function setup_paths()
%SETUP_PATHS  Add MATLAB and Python paths for scifor integration tests.
%
%   Delegates to the parent setup_paths to add common paths.

    parent_dir = fullfile(fileparts(mfilename('fullpath')), '..');
    run(fullfile(parent_dir, 'setup_paths.m'));
end
