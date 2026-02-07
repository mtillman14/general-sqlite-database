# Core API

::: thunk.core.thunk
options:
show_root_heading: true
show_source: false

::: thunk.core.Thunk
options:
show_root_heading: true
members: - **init** - **call** - hash - unpack_outputs - unwrap

::: thunk.core.PipelineThunk
options:
show_root_heading: true
members: - **init** - **call** - hash - inputs - outputs - is_complete - compute_cache_key

::: thunk.core.ThunkOutput
options:
show_root_heading: true
members: - **init** - data - hash - pipeline_thunk - output_num - is_complete - was_cached - cached_id

::: thunk.core.configure_cache
options:
show_root_heading: true

::: thunk.core.get_cache_backend
options:
show_root_heading: true

::: thunk.core.CacheBackend
options:
show_root_heading: true
