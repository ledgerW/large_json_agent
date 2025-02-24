from agents.tools import query_duckdb

# Test the query_duckdb function with the same query that was causing the error
result = query_duckdb("llamacloud", "SELECT table_name, column_name, data_type FROM information_schema.columns;")
print(result)
