# Plan — Variant Selection

> Concepts: `docs/claude/variant-selection.md`. Extends
> `.claude/plan-function-version-variants.md`.
>
> Status: **approved 2026-09-08. Stages 0-4a COMPLETE, ALL TESTS PASSING
> (user-run 2026-09-08), UNCOMMITTED. Stage 4b not started.**
>
>
> Stages 1 and 2 shipped together deliberately: chain columns without a
> chain-aware pin would give a table whose default pin collapses only one of
> several code axes — actively worse than either end state.

## Why

Two problems, one root. `CodeVersion` is a **one-hop** discriminator, so
multi-layer pipelines silently overplot records that differ only by upstream
code (Stage 1). And "compare method v1 against v3" is not expressible at all,
because supersession is the only intent the stack models and the old source is
discarded (Stages 3–4).

## Scope decisions already made (do not re-litigate)

- **`function_hash` does NOT join the supersession key.** It would destabilise
  `load`, node state and `find_record_id` together
  (`project_latest_record_selection_future_issue`). Supersession stays the
  default for a body edit; comparison becomes something the user *declares*.
- **Copy-paste-rename is rejected** as the method-comparison workflow. It
  duplicates what the DB already records and forks the DAG node. Parallel
  long-lived methods (Butterworth vs FIR) still get two names; iterative
  refinement gets one function with many recorded versions.
- **Plot variant selection must not touch `ParameterNode`'s checkboxes.** Those
  are execution state ("excludes unchecked values from multi-value fan-outs").
  Reuse the canvas *idiom*, not the canvas *instance*.
- **The chain walk lives in scidb** (CLAUDE.md NOTE 3). scistackplotdb consumes
  it; duplicating it there is the `feedback_avoid_scifor_scidb_duplication`
  mistake one layer up.

## Staging

Stages are independently valuable and independently shippable. **Stage 0 and
Stage 5 depend on nothing** and can land first.

### Stage 0 — Characterise the one-hop bug ✅ DONE, xfails flipped at Stage 1

Pure diagnosis, commits to no design. Per CLAUDE.md NOTE 2.

- `scistackplotdb/tests/test_variant_chain.py`.
- Fixture models the real sequence: run both layers → edit the **upstream**
  body → run both layers again. A single re-run is NOT enough — the downstream
  step consumes *latest* upstream, so two downstream records only appear once
  the pipeline is re-run end to end.
- One **passing** test for the precondition (two downstream records coexist at
  one location, sharing one producing `function_hash`).
- One **`xfail(strict=True)`** test for the desired behaviour (something
  distinguishes them / the pooling guard fires). Flips to pass at Stage 1.
- Adds `Summarized` to `scistackplotdb/tests/conftest.py` — a second
  pipeline-produced variable, needed for any two-layer test.

### Stage 1 — `code_versions_batch` (scidb) ✅ DONE, tests passing

- New batched read beside `branch_params_batch`, walking the same chain: one
  entry per **upstream function** holding more than one version.
- **Ordinals scoped per function**, not per variable type as `CodeVersion` is
  today. A function's v1/v2/v3 is globally coherent and sidesteps the scope
  argument `function-version-variants.md` had to work through.
- `variant_identity_batch` grows a `code_chain` field. Existing fields
  unchanged — the GUI variable panel must keep working untouched.
- Tests: `scidb/tests/test_variant_identity.py`, extending the existing
  two-version cases to two layers.

### Stage 2 — Chain-aware `CodeIsLatest` + column attachment ✅ DONE, tests passing

Deviation from the plan as written, decided during implementation: the single
`CodeVersion` column is **gone**, replaced by one `Code:<fn>` column per
multi-version function (`VERSION_FACTOR_PREFIX`). A merged column cannot express
a chain, and an adaptive name — `CodeVersion` at one function, per-function
names at two — would silently invalidate saved `PlotSpec`s the moment an
unrelated function was edited. Clean break, no alias
(`feedback_beta_no_deprecation`). `RAW_VERSION_LEVEL "(raw)"` became
`MISSING_VERSION_LEVEL "(n/a)"`, since per-function the question is "was this
function in my chain", not "am I a raw save".

Stage 5.2 (factor ordering) was folded in here rather than deferred — it is two
lines and the code-first ordering only makes sense alongside the new columns.

- `attach_variants` emits one column per chain member with >1 version.
- `CodeIsLatest` becomes "this record's whole upstream chain is the newest
  present **at its own schema location**". Per-location scoping is load-bearing
  (`load.py:37-46`) — a type-wide latest silently drops subjects never re-run.
- **Invariant to hold: the default stays one checkbox** however many layers
  generate versions. `roles.default_spec` already pins when `default_pin`
  exists; the pin must now collapse the whole chain.
- Fixes the `source.py:204-213` two-measure join, which currently drops the
  latest flag and falls back to showing every version. Tolerable at one hop,
  bad with a chain.
- Stage 0's xfail flips to pass here.

### Stage 3 — `_function_source` (scidb) ✅ DONE (Python only), tests passing

Resolved the open question **per-callee rows, not a flattened blob, and not
content-addressed dedup**: one row per unit keyed `(function_hash, unit_name)`,
with `is_entry` marking the function the hash is named for. A shared helper is
therefore stored once per calling hash. That duplication is deliberate — source
is kilobytes, dedup would cost a second table and a join on every read, and no
read path here is hot.

The drift hazard was closed by **not writing a second walk**:
`scilineage.hashing._hash_source` gained an optional `collect` dict, and
`compute_function_hash_with_sources` returns `(hash, units)` from that one
traversal. A separate source-walk that diverged from the hash-walk would file
units under a hash they did not correspond to.

`foreach_config.function_sources_for` is the capture recipe. It deliberately
does **not** reuse `function_hash_for`'s `.fcn` unwrap: the stored `__fn_hash`
comes from `to_version_keys` → `_compute_fn_hash(self.fn)`, which hashes the
object it was handed, so source must come from that same object. The write site
**verifies** derived hash == stored hash and refuses to write on mismatch —
source filed under the wrong key reads as captured and returns the wrong code,
which is worse than none.

⚠️ **MATLAB captures nothing yet.** `MatlabLineageFcn` carries `source_hash` but
not the text. `function_sources_for` duck-types a `source_text` attribute, so
closing this is a one-line bridge change plus a MATLAB-side plumb — but until
then MATLAB versions are still fossils. Not a silent gap: the write path logs it
at debug and `function_source` returns empty `units`, which callers must read as
"not captured", never "no code".

- New table keyed by `function_hash`. Purely additive: the hash is already the
  key, so no existing identity changes.
- Source is already in hand at hash time —
  `compute_matlab_function_hash(source_text, …)` is handed the text;
  `compute_function_hash` AST-hashes the live callable. Precedent for storing
  it: `GlueSpec.source_text`.
- ⚠️ Python's hash is **recursive over user callees**
  (`scilineage/hashing.py:185-188`), so a version needs the callee **closure**,
  not one string. Open: per-callee rows vs one flattened blob.
- Write-time capture (every invocation). Lazy capture would miss the baseline
  version entirely.
- Independently valuable even if Stage 4 never happens: it makes "what did the
  code that produced this record actually say?" answerable at all.

### Stage 4 — A declared version axis 🔶 REDESIGNED, gated on one decision

**Two findings from investigation on 2026-09-08 change this stage's shape. Read
before writing any code.**

**Finding 1 — the axis primitive already exists, and it is not `Version`.**
`scidb.Variant` (`scidb/variant.py`) already pins an input to a branch-param
variant as an **order-agnostic, load-time filter**, and its own docstring
already documents the declared-axis pattern:

```python
EachOf(Variant(FilteredEMG, low_hz=20), Variant(FilteredEMG, low_hz=50))
```

So the plan's `EachOf(Version("fn", "v1"), ...)` should instead become a **code
dimension on `Variant`** — the same wrapper, one more thing it can pin. Since
Stages 1–2 made `Code:<fn>` a variant column sitting beside branch params, this
is not a hack: they are one concept and deserve one filter. Suggested surface,
mirroring the existing `fn=` disambiguation:

```python
Variant(FilteredEMG, code_version="v1")                  # unambiguous upstream
Variant(FilteredEMG, fn="bandpass", code_version="v1")   # disambiguated
Variant(FilteredEMG, code_version="latest")              # the common default
```

Implementation is small: carry it as a reserved key in the existing filter dict
(the precedent is `__save__.<kwarg>`), and teach
`database._filter_records_by_branch_params` to route that key through
`code_versions_batch` (Stage 1) instead of `branch_params_batch`. **No new
plumbing** through the ~10 `branch_params_filter` call sites, and **no exec of
stored source**.

This also splits the stage cleanly:

- **4a — selection** among already-computed records. Serves the actual question
  that started this work ("how do I pick one variant out of the mess?"), needs
  no Stage 3, no `exec`.
- **4b — re-execution** of an old version from `_function_source`. The expensive,
  genuinely risky half. Nothing needs it yet.

**Finding 2 — the "pinned node reads needs-run" bug is PRE-EXISTING and already
affects branch-param pins.** It is not specific to code versions:

- `config_from_inputs` unwraps `Fixed`/`ColumnSelection` to reach a `type`, but
  has no `Variant` branch — a `Variant` instance is not a `type`, so the param
  never lands in `input_types`.
- `_predict_config_invocations` enumerates `_current_records_by_schema`, which
  is latest-per-(location, producing-variant): **one record per variant, all of
  them.** The pin is nowhere in sight.

So a node pinned to one variant is predicted to owe invocations over the variant
it was explicitly told to skip → permanently needs-run. Characterised by
`scidb/tests/test_variant_pin_node_state.py` (`xfail(strict=True)`).

✅ **DECIDED 2026-09-08: do none of them. Leave node state fully derived.**

The user's objection settles it, and it is the right one: today **both sides of
the completeness comparison are derived** from the same append-only,
content-addressed graph. Nothing is stored that could disagree with anything
else, so node colour can be wrong only if the derivation is wrong — never
because something went stale.

Persisting the declaration (option 1, which I recommended twice — wrongly) gives
that up in the dangerous direction:

> Run with `Variant(Scaled, factor=2.0)`; the pin is stored. Later edit the
> script to drop the pin. The code now says "run over everything", the database
> still says "pinned". The node reads **green** while `factor=3.0` has never
> run.

Falsely green is much worse than falsely red, and editing a `for_each` call is
the most ordinary thing anyone does to a pipeline.

**If this is ever fixed, use option 3 (live declaration only)** — read the pin
from the current source each time, store nothing, stay unstaleable. Note it is
not free: `inputs_fallback` currently *adds* predictions on top of the
graph-derived configs, so a live pin would have to **constrain** those configs
rather than supplement them.

**Do not re-propose persistence.** The gain does not justify it: a pinned node
showing red is cosmetic — no data is wrong, no results are affected — it is a
pre-existing bug in a pre-existing feature (branch-param pinning shipped with
it), and it is unrelated to the plotting question that started this work.

The three options are kept below only so the reasoning is not re-derived.

⛔ ~~The blocking decision, which is a product question and not mine to make.~~
For an *already-run* config, `function_variant_configs` reconstructs from the
graph — and from the graph, "pinned to low_hz=20" and "has not got round to
low_hz=50 yet" look identical. Both are "only low_hz=20 records were consumed".
Distinguishing them needs one of:

1. **Persist the declaration.** Store the pin (as part of the config / call
   site) so the graph records intent, not just outcome. Most correct; touches
   the save path.
2. **Infer from consistency.** If *every* realized invocation of a config
   consumed one variant, treat the config as pinned. No schema change; wrong the
   first time a genuinely-incomplete node happens to be consistent so far.
3. **Only honour a live declaration.** Use the pin when `inputs_fallback` is
   supplied (the GUI knows the declared inputs) and accept that a graph-only
   query cannot tell. Smallest; leaves the answer dependent on who is asking.

Pick one before implementing 4a's node-state half. 4a's *filtering* half is
independent of this and safe to build first.

#### Stage 4a — filtering ✅ DONE, tests passing

`Variant(X, code_version=…, fn=…)`, carried as a reserved `__code__` /
`__code__.<fn>` key in the existing filter dict and resolved by
`database._filter_records_by_code_version` through Stage 1's
`code_versions_batch` / `code_version_ordinals`. Zero new plumbing; no `exec`.

Three semantics worth not re-deriving:

- A **named ordinal** (`"v1"`) is per-function and global — it *does* drop schema
  locations that never ran that version. Correct: the user named it.
- **`"latest"`** is resolved per *location* via `variant_identity_batch`'s
  chain-wide `is_latest`, so nothing drops out. Not a synonym for "highest
  ordinal".
- A pin on a project where **nothing is versioned is a no-op**, selecting
  everything rather than nothing. An empty run is a silent disaster; an ignored
  pin is not.

**The pin is per-dimension, and a qualified pin is therefore PARTIAL.** Raised
by the user 2026-09-08. The *lookup* is not one-hop — `code_versions_batch`
walks the whole chain, so `fn=` can name any ancestor — but with two versioned
functions upstream, `fn="bandpass", code_version="v1"` keeps *both* `loadEMG`
versions. That is legitimate and is exactly the "all variants where x=1, y=2"
case, but it is not what "pin the variant" sounds like. `_filter_records_by_code_version`
now logs a PARTIAL warning naming the dimensions still free. `"latest"` is
chain-wide and never partial.

Investigating that found a **real bug, since fixed**: the `fn=`-qualified path
had none of the bare path's guards. `code_version_ordinals` omits single-version
functions, so `ordinals.get(fn_name, {})` was `{}` and every record failed the
comparison — pinning a function that had never been edited **silently emptied
the run**. Now: an unknown function name, an unknown ordinal, and `v2` on a
single-version function all raise and name what IS available; `v1` on a
single-version function is a satisfied pin, because its one version is v1.

⚠️ **A code pin must load UNCOLLAPSED, and missing this made the whole feature a
no-op at first.** `_load_input` loads with `version_id="latest"`, which collapses
on `(fn_name, branch_params, output_num, consumed_locations)` — `function_hash`
is deliberately absent, because a body re-run is a newer version of the same
variant rather than a rival. So **both code versions merge and the newer wins
before any filter runs**, and pinning `v1` matched nothing. A branch-param pin
was never affected: `branch_params` IS in that key, so its variants never merged
in the first place. That asymmetry is the reason this looked like it should work.

`_load_var_type_as_spread` now passes `version_id="all"` **when and only when**
the filter carries a code pin, so an unpinned load is byte-identical to before.
Known limitation of skipping the collapse: if one version was saved twice at a
location (a re-run with no edit), both records survive the pin and the consumer
fans out over both. Not exercised by any test; fix by re-collapsing within the
surviving set if it ever bites.

Still open: node state (the decision above) — a code-pinned node inherits the
same permanent-needs-run bug as a branch-param-pinned one, since neither reaches
`_predict_config_invocations`.

#### Stage 4b — re-execution ⬜ not started

Materialising an old version from `_function_source` and running it. Nothing
needs it yet; 4a covers the question that started this work.

- `EachOf(Version("fn", "v1"), Version("fn", "v3"))` over recorded versions of
  one function. One DAG node, no duplication.
- Requires Stage 3 to be **re-runnable**; without it, still useful for
  *selecting among already-computed* records, which is most of the plotting
  value.
- ⚠️ **The hardest open question in the whole plan**: a node deliberately
  pinned to an OLD version must not read as needs-run.
  `expected_invocations_for_function` predicts from `fn_hash`, so a version axis
  means two expected sets per node. **Resolve this before committing to
  Stage 4's size** — it is the difference between a week and a month.
- Separately: `EachOf` over plain callables (the parallel-methods case) is cheap
  and independent of all of the above.

### Stage 5 — GUI

Ordered by value-per-unit-work, best first:

1. **Combination readout** — *"4 of 24 combinations · 12 panels"*. Depends on
   nothing; `plan_layout` already computes panel counts. Ship first.
2. **Factor ordering** — `source.py:247-249` puts schema keys before variants;
   `load.py` appends `CodeVersion` after the branch params (~line 259). Flip
   both: code version first, then parameter variants, then schema keys.
3. **Always emit `CodeVersion`**, even at one level, so the control has a
   permanent home. Safe: `default_roles` and `validate` both guard on
   `len(f.levels) > 1`.
4. **Version list on `FunctionNode`**, mirroring `ParameterNode`'s existing
   checkboxed value list. Closes the asymmetry between the two variant sources.
5. **Plot-scoped variant subgraph** in Plot Studio — induced subgraph on
   ancestors of the plotted measure contributing >1 level (usually 2–5 nodes).
   Bound to `PlotSpec`, never to the canvas's execution state.

⚠️ Any frontend change is dead until **both** vite bundles are rebuilt
(`project_frontend_bundle_rebuild`).

## Test commands

One package per invocation — bare `from conftest import` collides otherwise
(`project_pytest_one_package_at_a_time`).

```
pytest scistackplotdb/tests
pytest scidb/tests/test_variant_identity.py
pytest scistackplot/tests
```
