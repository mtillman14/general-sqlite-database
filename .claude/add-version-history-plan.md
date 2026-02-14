Plan: Batch Loading API with version_id Parameter

Context

After implementing the parameter_id + version_id scheme, we need a clean API for batch-loading across multiple parameter sets and version histories. The core problem: load(threshold=[0.3, 0.5]) is
ambiguous — does the list mean "match any" or is it the literal value?

Solution: load() treats all values as scalar literals. load_all() treats list values as "match any". Both use version_id to control which versions are returned.

API Design

Disambiguation Rule

- load() — scalar context. threshold=[0.3, 0.5] means the parameter IS the list [0.3, 0.5]. Returns one result (latest version). version_id is implicit "latest" (not user-facing).
- load_all() — batch context. threshold=[0.3, 0.5] means "threshold is 0.3 OR 0.5". Returns the full cartesian product of (matching schema_ids × matching parameter_ids × matching version_ids).

version_id parameter (replaces latest_only)

- version_id="all" (default for load_all()) — return all versions
- version_id="latest" (used internally by load()) — return only the row with max version_id per (parameter_id, schema_id)
- version_id=3 — return only version_id=3
- version_id=[1, 3] — return version_ids 1 and 3

Using strings "latest" / "all" instead of None avoids MATLAB-Python None translation issues.

Result count

The number of results from load_all() equals the number of elements in the cartesian product:

- schema locations matching the schema kwargs (or all if none specified)
- × parameter_ids matching the parameter kwargs (or all if none specified)
- × version_ids matching the version_id argument (or all if "all")

---

Changes by File

1.  src/scidb/database.py

a. \_find_record() — replace latest_only with version_id, add list-value support

def \_find_record(
self,
table_name: str,
record_id: str | None = None,
nested_metadata: dict | None = None,
version_id: int | list[int] | str = "latest", # replaces latest_only
) -> pd.DataFrame:

Schema key filtering — support lists with SQL IN:
for key, value in schema_keys.items():
if isinstance(value, (list, tuple)):
placeholders = ", ".join(["?"] \* len(value))
conditions.append(f's."{key}" IN ({placeholders})')
params.extend([str(v) for v in value])
else:
conditions.append(f's."{key}" = ?')
params.append(str(value))

Version key filtering — support lists with Python in:
for key, value in version_keys.items():
if isinstance(value, (list, tuple)):
mask = df["version_keys"].apply(
lambda vk, k=key, vals=value: json.loads(vk).get(k) in vals ...
)
else:
mask = df["version_keys"].apply(
lambda vk, k=key, v=value: json.loads(vk).get(k) == v ...
)

version_id filtering:

- "latest" → use CTE with ROW_NUMBER() OVER (PARTITION BY ... ORDER BY rm.version_id DESC) (same as old latest_only=True)
- "all" → no version filtering
- int → WHERE rm.version_id = ?
- list[int] → WHERE rm.version_id IN (...)

b. DatabaseManager.load() — update \_find_record call

# Change from:

records = self.\_find_record(table_name, nested_metadata=nested_metadata, latest_only=True)

# To:

records = self.\_find_record(table_name, nested_metadata=nested_metadata, version_id="latest")

c. DatabaseManager.load_all() — replace latest_only with version_id

def load_all(
self,
variable_class: Type[BaseVariable],
metadata: dict,
version_id: int | list[int] | str = "all", # replaces latest_only; "all" returns every version
):
Pass version_id through to \_find_record(). Note: callers that want latest-only (like load()) pass version_id="latest" explicitly.

2.  src/scidb/variable.py

a. BaseVariable.load() — update call

# Change from:

results = list(db.load_all(cls, metadata, latest_only=True))

# To:

results = list(db.load_all(cls, metadata, version_id="latest"))

b. BaseVariable.load_all() — add version_id parameter

@classmethod
def load_all(
cls,
as_df: bool = False,
include_record_id: bool = False,
version_id: int | list[int] | str = "all", # NEW — default "all" returns every version
\*\*metadata,
):
Pass version_id through to db.load_all(). Also update \_load_all_generator to accept and pass through version_id.

3.  scidb-matlab/src/scidb_matlab/matlab/+scidb/BaseVariable.m

a. load() method — update kwarg

% Change from:
py_gen = py_db.load_all(py_class, py_metadata, pyargs('latest_only', true));
% To:
py_gen = py_db.load_all(py_class, py_metadata, pyargs('version_id', 'latest'));

b. load_all() method — extract and pass version_id

Extract version_id from varargin (similar to split_version_arg pattern), default to "all", pass as pyarg to py_db.load_all().

4.  scidb-matlab/src/scidb_matlab/matlab/+scidb/+internal/ — helpers

a. New: split_load_all_args.m

Extract version_id from varargin, return remaining metadata args.

b. Update: metadata_to_pydict.m

Handle non-scalar arrays/string arrays as Python lists for batch matching:
if isnumeric(val) && ~isscalar(val)
py_dict{key} = py.list(num2cell(val));
elseif isstring(val) && ~isscalar(val)
py_dict{key} = py.list(cellfun(@char, num2cell(val), 'UniformOutput', false));
end

5.  Test Updates

a. sciduck/tests/test_sciduck.py — no changes needed (SciDuck layer unaffected)

b. tests/test_integration.py — add batch loading tests

- Test list-valued schema keys: load_all(subject=[1, 2])
- Test list-valued version keys: load_all(algorithm=["v1", "v2"])
- Test version_id parameter: load_all(version_id="all"), load_all(version_id=2)
- Test cartesian product: list schema keys x list version keys x version_id

c. scidb-matlab/tests/matlab/TestSaveLoad.m — no changes needed

Default version_id="all" preserves current behavior (all versions returned).

d. scidb-matlab/tests/matlab/TestEndToEnd.m — no changes needed

Default version_id="all" preserves current behavior.

---

Breaking Changes

1.  latest_only parameter removed from DatabaseManager.load_all() — replaced by version_id.
2.  load_all() default is version_id="all" which preserves current behavior (returns all versions). Not a breaking change for end users.

Implementation Order

1.  Update \_find_record() in database.py (replace latest_only, add list-value support, add version_id filtering)
2.  Update DatabaseManager.load() and load_all() signatures
3.  Update BaseVariable.load() and load_all() in variable.py
4.  Update MATLAB bridge (BaseVariable.m load/load_all, new helper, metadata_to_pydict)
5.  Update tests (Python + MATLAB)

Verification

1.  python -m pytest tests/ -v — Python integration tests
2.  python -m pytest sciduck/tests/ -v — SciDuck tests (should be unaffected)
3.  Key scenarios:

- load_all(subject=[1,2], smoothing=[0.2, 0.3]) returns all combinations (cartesian product)
- load_all() with no version_id → returns all versions (default "all")
- load_all(version_id="latest") → returns latest per parameter set
- load_all(version_id=2) → returns only version 2
- load_all(version_id=[1, 3]) → returns versions 1 and 3
- load(threshold=[0.3, 0.5]) matches literal list value (scalar context)
