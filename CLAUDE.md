This is a software package to facilitate scientific data processing. You should learn more about it by reading the README.md

There are several submodules in this package. Here are the folders that they are in, and their general purpose within the package:

- pipelinedb-lib: SQLite lineage database layer
- scidb-matlab: MATLAB wrapper
- scidb-net (optional)
- sciduck: DuckDB database layer
- scirun-lib: The highest level user-facing abstractions for batch run/essential convenience operations
- src/ (scidb): The root folder package, the core user-facing abstractions
- thunk-lib: Lineage package

Each package's folder has a README.md file. When you go to look for implementation details, please start by reading the relevant README.md. If you have sufficient information after that, then please answer the question without exploring additional unnecessary files.

The next place to look for context is the docs/claude folder. This folder contains documentation that was written by you previously, specifically to fill conceptual gaps. If you don't find sufficient information there, then look through the integration tests in each package's tests/ folder. If you have sufficient information after that, then please answer the question without exploring additional unnecessary files. Otherwise, look through the relevant source code.

Finally, after you've collected all relevant information (by reading through the README's and stopping, or by then reading through the docs/claude folder and stopping, or by then reading through the tests and stopping, etc.), please always ask the user if they would like to pause and write a file to docs/claude to fill conceptual gaps that you can look at later to better understand that aspect of the code's function.

Also, every time you draft a plan and present it to me for approval, please also write a .claude/plan-name.md file.
