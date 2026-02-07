# Core API

::: thunk.core.thunk
options:
show_root_heading: true
show_source: false

::: thunk.core.Thunk
options:
show_root_heading: true
members: - **init** - **call** - hash - unpack_output - unwrap - query

::: thunk.core.PipelineThunk
options:
show_root_heading: true
members: - **init** - **call** - hash - inputs - outputs - is_complete - compute_lineage_hash

::: thunk.core.ThunkOutput
options:
show_root_heading: true
members: - **init** - data - hash - pipeline_thunk - output_num - is_complete
