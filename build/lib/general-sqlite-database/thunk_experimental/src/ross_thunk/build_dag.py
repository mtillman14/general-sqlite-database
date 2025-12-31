from base_dag import DAG
from ross_thunk.thunk import Thunk, OutputThunk
from ross_thunk.constant import Constant

def build_dag(thunks: tuple[Thunk]) -> DAG:
    """Builds a DAG from a list of Thunk instances."""
    dag = DAG()

    # Get all of the PipelineThunks
    pipeline_thunks = []
    for thunk in thunks:
        pipeline_thunks.extend(thunk.pipeline_thunks)

    # Create edges based on inputs and outputs
    for pipeline_thunk in pipeline_thunks:
        dag.add_node(pipeline_thunk)

        # Add edges based on outputs
        for output_thunk in pipeline_thunk.outputs:
            dag.add_edge(pipeline_thunk, output_thunk)

        # Add edges based on inputs
        for input_value in pipeline_thunk.inputs.values():
            if isinstance(input_value, OutputThunk):
                dag.add_edge(input_value, pipeline_thunk)
            else:
                # If the input is a constant, wrap it in a Constant node
                constant_node = Constant(input_value)                
                dag.add_edge(constant_node, pipeline_thunk)

    return dag