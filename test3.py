import duckdb                                                                                                                                                                                       
                                                                                                                                                                                                     
# Create an in-memory DuckDB database
con = duckdb.connect(':memory:')

# Create a table with VARCHAR columns
con.execute("""
    CREATE TABLE _schema (
        schema_id INTEGER PRIMARY KEY,
        trial VARCHAR
    )
""")

# Insert values as strings (as the system does)
con.execute("INSERT INTO _schema VALUES (1, '1')")
con.execute("INSERT INTO _schema VALUES (2, '2')")
con.execute("INSERT INTO _schema VALUES (3, '10')")
con.execute("INSERT INTO _schema VALUES (4, '3')")

# Query with ORDER BY (what distinct_schema_values does)
rows = con.execute("SELECT DISTINCT trial FROM _schema WHERE trial IS NOT NULL ORDER BY trial").fetchall()
print("Results from DuckDB ORDER BY on VARCHAR:")
for row in rows:
    print(f"  Value: {row[0]!r}, Type: {type(row[0])}")