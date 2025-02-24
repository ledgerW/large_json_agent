import os
from dotenv import load_dotenv
load_dotenv()

import json
import orjson
from pandas import json_normalize
import duckdb
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from typing import Annotated


# Load model
model = ChatOpenAI(model="gpt-4o")


def query_json(query: str, json_file_path: str) -> str:
    """
    Query JSON data using DuckDB with each top-level section loaded as a separate table.
    
    This function loads the JSON file using orjson for speed, creates separate tables for
    each top-level section, and allows querying across these tables using standard SQL.
    
    Args:
        query: SQL query to execute on the tables
        json_file_path: Path to the JSON file to query
        
    Returns:
        The query results as a string
    """
    try:
        # Check if file exists
        if not os.path.exists(json_file_path):
            return f"Error: File {json_file_path} not found"

        # Read JSON with orjson for better performance
        with open(json_file_path, 'rb') as f:
            data = orjson.loads(f.read())
        
        # Create DuckDB connection
        con = duckdb.connect(':memory:')
        
        # Load each top-level section as a separate table
        for section, section_data in data.items():
            print(f"Processing section: {section}")
            
            # Handle both single object and list of objects
            if isinstance(section_data, dict):
                section_data = [section_data]
            elif not isinstance(section_data, list):
                print(f"Skipping section {section} - not a dict or list")
                continue
                
            try:
                # Use pandas json_normalize to flatten nested structures
                df = json_normalize(section_data, sep='.')
                
                # Register DataFrame as a table with the section name
                con.register(section, df)
                print(f"Successfully registered table: {section}")
            except Exception as e:
                print(f"Error processing section {section}: {str(e)}")
                continue
        
        # Execute query
        result = con.execute(query).df()
        
        # Convert result to string representation
        if len(result) == 0:
            return "No results found"
        return result.to_string()
        
    except Exception as e:
        return f"Error querying JSON: {str(e)}"


@tool("query_json_data")
def query_json_tool(
    query: Annotated[str, """The DuckDB SQL query to execute against the available tables. Uses standard SQL syntax."""],
    json_file_path: Annotated[str, "Path to the JSON file to query"]
) -> Annotated[str, "Query results as a string"]:
    """
    Use this tool to query the JSON data file using SQL via DuckDB.
    
    The JSON data is automatically loaded with each top-level section as a separate table.
    Nested objects and arrays within each section are expanded into columns using dot notation.
    Arrays are properly maintained as lists where appropriate.
    
    Available tables correspond to the top-level sections in the JSON:
    - jobs
    
    You can query across multiple tables using standard SQL JOIN operations.
    """
    return query_json(query, json_file_path)


system_prompt = """
You are an expert JSON data analyst specializing in SQL queries over complex nested JSON data structures.
You have access to a JSON schema and a JSON data file, which you'll use to answer questions about the data.

The data is automatically loaded into separate tables for each top-level section where:
- Each top-level section (metadata, users, products, transactions, logs, analytics) is a separate table
- Nested objects are flattened using dot notation (e.g., profile.name)
- Arrays are preserved as queryable lists
- Complex nested structures are maintained in a way that enables sophisticated queries

You must always:
1. Use SQL to understand the schema first by running:
   - SELECT * FROM information_schema.columns to get detailed column info for all tables
   - DESCRIBE or SHOW COLUMNS to examine available fields for specific tables
   - SELECT * FROM <table_name> LIMIT 1 to see sample data structure
   - Check array fields with ARRAY_LENGTH() where relevant
2. Write efficient and precise SQL queries that match the data structure
3. Utilize DuckDB's advanced features for complex operations

Example Queries (from simple to complex):

Basic Queries:
- Simple field selection:
  SELECT name, email 
  FROM users

- Filtering with nested fields:
  SELECT name, profile.preferences.theme
  FROM users 
  WHERE profile.preferences.theme = 'dark'

Intermediate Queries:
- Counting nested arrays:
  SELECT json_array_length(json_extract(orders, '$')) AS number_of_orders
  FROM users;

- Array operations:
  SELECT 
    name,
    json_extract(orders, '$[0].date') as first_order_date,
    json_array_length(json_extract(orders, '$')) as total_orders
  FROM users

- Array unnesting with aggregation:
  SELECT u.name, ARRAY_LENGTH(u.orders) AS total_orders
  FROM (
    SELECT UNNEST(data.users) AS u
    FROM json_data AS data
    ) AS unnested_users
  WHERE u.name = 'John Doe';


Advanced Queries:
- Cross-table queries:
  SELECT 
    u.name,
    t.transaction_id,
    p.product_name
  FROM users u
  JOIN transactions t ON t.user_id = u.id
  JOIN products p ON p.id = t.product_id

- Complex array manipulations with analytics:
  SELECT 
    u.name,
    COUNT(CASE WHEN o.status = 'completed' THEN 1 END) as completed_orders,
    ARRAY_AGG(DISTINCT i.product_id) as purchased_products,
    a.user_engagement_score
  FROM users u
  LEFT JOIN UNNEST(u.orders) AS o
  LEFT JOIN UNNEST(o.items) AS i
  LEFT JOIN analytics a ON a.user_id = u.id
  GROUP BY u.name, a.user_engagement_score

Remember to:
- The JSON data is loaded into separate tables for each top-level section
- Leverage DuckDB's full SQL capabilities including window functions, CTEs, and subqueries
- Write queries that handle NULL values and edge cases appropriately
- Use appropriate array functions (UNNEST, ARRAY_AGG, ARRAY_LENGTH) for nested data
- Consider performance implications when dealing with large nested structures

Available tables correspond to the top-level sections in the JSON:
- jobs
"""

# The Schema Agent
large_json_agent = create_react_agent(
    model=model,
    tools=[query_json_tool],
    name="large_json_analyst",
    prompt=system_prompt
)