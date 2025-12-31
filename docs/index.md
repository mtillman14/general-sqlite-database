# General SQLite Database for Scientific Computing

Syntax:
```python
from dataset_builder import Dataset
from database import Database
from variable import Variable

ds = Dataset("config.toml")
db = Database(ds, db_file_path="db.sqlite")
# Example usage to load data
for trial in db.trials:
    data = load_data(trial) # Dummy function to represent data loading
    data_var = Variable(data=data, data_object=trial)
    db.save(data_var)
```