import uuid
from datetime import datetime

from src.ross_thunk.thunk_persistence.ThunkDatabase import ThunkDatabase

db = ThunkDatabase("thunks.db")

# Example PipelineThunk data
pipeline_data = {
    'id': str(uuid.uuid4()),
    'func_name': 'process_pipeline',
    'args': (1, 2, 3),
    'kwargs': {'batch_size': 32},
    'result': [4, 5, 6],
    'execution_time': 0.123,
    'timestamp': datetime.now()
}

# Example OutputThunk data with complex value
output_data = {
    'id': str(uuid.uuid4()),
    'func_name': 'generate_output',
    'args': ('input.txt',),
    'kwargs': {'format': 'json'},
    'result': {'status': 'success'},
    'value': {  # Complex value that needs special storage
        'data': [1, 2, 3, 4, 5],
        'metadata': {'version': '1.0', 'created': '2024-01-01'},
        'binary_data': b'some binary content'
    },
    'execution_time': 0.456,
    'timestamp': datetime.now()
}

# Write different types of thunks
db.write_pipeline_thunk(pipeline_data)
db.write_output_thunk(output_data)

print("Reading thunks back...")

# Read them back
retrieved_pipeline = db.get_pipeline_thunk(pipeline_data['id'])
retrieved_output = db.get_output_thunk(output_data['id'])

print(f"Pipeline thunk: {retrieved_pipeline}")
print(f"Output thunk: {retrieved_output}")

print("\n=== Demonstrates Different Storage Strategies ===")
print("PipelineThunk: Simple JSON storage, same across databases")
print("OutputThunk: Complex value storage, different strategies per database")
print("- SQLite: Separate columns for different value types")
print("- PostgreSQL: Native JSONB for complex values")