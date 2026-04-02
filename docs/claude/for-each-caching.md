# for_each Caching / skip_computed

See `.claude/skip-computed-plan.md` for the full design. This document
summarises the key concepts.

## Where it lives

`scihist.for_each` — not `scidb.for_each`. scidb is a generic iteration engine
with no lineage awareness; the skip logic belongs in scihist.

## API

```python
scihist.for_each(fn, inputs=..., outputs=..., skip_computed=True, ...)
```

`skip_computed=True` is the default.

## How staleness is detected (no timestamps)

For each schema combo, before loading inputs:

1. **Output exists?** If not → compute.
2. **Walk the full upstream graph** via `db.get_upstream_provenance(output_record_id)`.
3. For each upstream node:
   - **Function hash**: compare `_lineage.function_hash` to current `LineageFcn.hash`.
   - **Input record_ids**: `_lineage.inputs` stores the exact `record_id` of
     each input variable used at save time. Compare each to the current
     `_find_record(..., version_id="latest")` result. Mismatch → upstream changed.
4. All checks pass → skip and print `[skip]`. Any failure → print `[recompute]`
   with the reason.

`record_id` is content-addressed (derived from content hash + metadata), so
identical data always produces the same `record_id` regardless of when it was
saved. This makes the comparison stable.

## The cascade

`get_upstream_provenance` gives the full pipeline graph. `run.py` uses it to
run functions in topological order, each with `skip_computed=True`. Each
function's skip check sees freshly-updated intermediate outputs and
correctly re-runs or skips — no function needs to look beyond its own
`_lineage.inputs`.

## "Latest version" semantics

When looking up the current record for an input, the skip check uses
`_find_record(..., version_id="latest")` — the same timestamp-based partition
logic used by every `load()` call in scidb. No new timestamp logic is added.
