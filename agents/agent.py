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


# === Schema Agent and Functions ===
def load_json_schema_section(schema_path: str, section_path: str) -> str:
    """Load any section or subsection of the large JSON schema using dot notation."""
    with open(schema_path, "r") as f:
        schema = json.load(f)
    
    # Handle empty/root path
    if not section_path:
        return json.dumps(schema, indent=2)
        
    # Navigate through the path
    current = schema
    path_parts = section_path.split('.')
    
    for part in path_parts:
        # Handle array items
        if isinstance(current, dict) and current.get('type') == 'array':
            current = current.get('items', {})
            
        # Navigate to next level
        if isinstance(current, dict):
            current = current.get('properties', {}).get(part, {})
        else:
            return json.dumps({"error": f"Invalid path: {section_path}"}, indent=2)
            
    return json.dumps(current, indent=2)


@tool("view_schema_section")
def view_schema_section(
    section_path: Annotated[str, "A dot-separated path string like 'users.profile.security' where each segment represents a nested JSON object. Example: 'transactions.fraud_check.flags'"], 
    schema_path: Annotated[str, "The path to the JSON schema file"]
) -> Annotated[str, "The section of the schema as a JSON string"]:
    """
    View any section or subsection of the JSON schema by providing a dot-notation path.
    
    This tool allows you to inspect specific parts of the schema structure by navigating through nested objects.
    The path should be provided using dot notation where each segment represents a nested object level.
    
    Example paths:
    - '' (empty string) - Returns the complete schema
    - 'users' - Returns the users section schema
    - 'users.profile' - Returns the profile subsection under users
    - 'users.profile.security' - Returns the security subsection under profile
    - 'transactions.fraud_check' - Returns the fraud check section of transactions
    
    The tool will return the requested section as a formatted JSON string, making it easy to examine
    the schema requirements, data types, and constraints for that specific part of the data structure.
    """
    return load_json_schema_section(schema_path, section_path)


@tool("view_available_sections")
def view_available_sections(
    schema_path: Annotated[str, "The path to the JSON schema file"],
    section_path: Annotated[str, "Optional dot-separated path to view subsections (e.g. 'users.profile')"] = ""
) -> Annotated[str, "A list of available sections in the JSON schema"]:
    """
    View available sections and subsections in the JSON schema.
    
    This tool helps explore the schema structure by listing all available sections at a given level.
    If no section_path is provided, it shows top-level sections.
    If a section_path is provided (e.g. 'users.profile'), it shows subsections at that path.
    
    Example paths:
    - "" (empty string) - Shows top-level sections like users, products, transactions
    - "users" - Shows subsections under users like profile, addresses, orders
    - "users.profile" - Shows subsections under profile like preferences, security
    """
    with open(schema_path, "r") as f:
        schema = json.load(f)
    
    # Navigate to the specified section if provided
    current = schema
    if section_path:
        try:
            for part in section_path.split('.'):
                current = current.get('properties', {}).get(part, {})
                # Handle array items
                if current.get('type') == 'array':
                    current = current.get('items', {})
        except (KeyError, AttributeError):
            return f"Invalid section path: {section_path}"
            
    # Get available properties at current level
    properties = current.get('properties', {}).keys()
    if not properties:
        return "No subsections available at this path"
        
    path_prefix = f"{section_path}." if section_path else ""
    sections_list = [f"{path_prefix}{prop}" for prop in properties]
    return f"Available sections: {', '.join(sections_list)}"


def query_json(query: str, json_file_path: str, section: str = "") -> str:
    """
    Query a specific section of a JSON file using DuckDB with flattened data structure.
    
    This function loads the JSON file using orjson for speed, extracts the specified section,
    normalizes the nested structure using pandas json_normalize, and then queries it using DuckDB.
    
    Args:
        query: SQL query to execute on the flattened data
        json_file_path: Path to the JSON file to query
        section: Top-level section key to query (empty string means query entire file)
        
    Returns:
        The query results as a string
    """
    try:
        # Read JSON with orjson for better performance
        with open(json_file_path, 'rb') as f:
            data = orjson.loads(f.read())
        
        # Extract specific section if provided
        if section:
            if section not in data:
                return f"Error: Section '{section}' not found in JSON data"
            data = data[section]
            
        # Handle both single object and list of objects
        if isinstance(data, dict):
            data = [data]
            
        # Use pandas json_normalize to flatten nested structures
        df = json_normalize(data, sep='.')
        
        # Create DuckDB connection and register DataFrame
        con = duckdb.connect()
        con.register('json_data', df)
        
        # Execute query
        result = con.execute(query).df()
        
        # Convert result to string representation
        return result.to_string()
        
    except Exception as e:
        return f"Error querying JSON: {str(e)}"


@tool("query_json_data")
def query_json_tool(
    query: Annotated[str, """The SQL query to execute against the flattened JSON data. Uses standard SQL syntax."""],
    json_file_path: Annotated[str, "Path to the JSON file to query"],
    section: Annotated[str, "Optional top-level section key to query (e.g., 'users', 'transactions'). Leave empty to query entire file."] = ""
) -> Annotated[str, "Query results as a string"]:
    """
    Use this tool to query a specific section of the JSON data file using SQL via DuckDB.
    
    The JSON data is automatically flattened with nested objects and arrays expanded
    into columns using dot notation. Arrays are properly maintained as lists where appropriate.
    
    You can specify a top-level section to query, which helps focus the analysis on a specific
    part of the data. For example:
    - section="users" will only query the users array/object
    - section="transactions" will only query the transactions section
    - empty section will query the entire file
    
    For example, a nested structure in the users section like:
    {
        "users": {
            "name": "John",
            "addresses": [
                {"type": "home", "city": "NY"},
                {"type": "work", "city": "SF"}
            ]
        }
    }
    
    becomes columns like:
    name, addresses
    
    Where addresses remains a proper array that can be queried using DuckDB's array functions.
    
    Example queries:
    - Basic nested field: SELECT name FROM json_data
    - Array operations: SELECT addresses[0].city FROM json_data
    - Array unnesting: SELECT UNNEST(addresses).city FROM json_data
    
    Use standard SQL syntax along with DuckDB's array handling functions when needed.
    """
    return query_json(query, json_file_path, section)




system_prompt = """
You are an expert JSON data analyst specializing in SQL queries over complex nested JSON data structures.
You have access to a JSON schema and a JSON data file, which you'll use to answer questions about the data.

The data is automatically normalized into a SQL-queryable format where:
- Nested objects are flattened using dot notation (e.g., user.profile.name)
- Arrays are preserved as queryable lists
- Complex nested structures are maintained in a way that enables sophisticated queries

You must always:
1. Use SQL to understand the schema first by running:
   - SELECT * FROM information_schema.columns WHERE table_name = 'json_data' to get detailed column info
   - DESCRIBE or SHOW COLUMNS to examine available fields
   - SELECT * FROM json_data LIMIT 1 to see sample data structure 
   - Check array fields with ARRAY_LENGTH() where relevant
2. Write efficient and precise SQL queries that match the data structure
3. Utilize DuckDB's advanced features for complex operations

Example Queries (from simple to complex):

Basic Queries:
- Simple field selection:
  SELECT u.name, u.email 
  FROM (SELECT UNNEST(data.users) AS u FROM json_data AS data) AS users

- Filtering with nested fields:
  SELECT u.name, u.profile.preferences.theme
  FROM (SELECT UNNEST(data.users) AS u FROM json_data AS data) AS users 
  WHERE u.profile.preferences.theme = 'dark'

Intermediate Queries:
- Array operations:
  SELECT 
    u.name,
    u.orders[0].date as first_order_date,
    ARRAY_LENGTH(u.orders) as total_orders
  FROM (SELECT UNNEST(data.users) AS u FROM json_data AS data) AS users

- Array unnesting with aggregation:
  SELECT 
    u.name,
    COUNT(*) as total_items
  FROM (SELECT UNNEST(data.users) AS u FROM json_data AS data) AS users,
       UNNEST(u.orders) AS o
  GROUP BY u.name

Advanced Queries:
- Complex array manipulations:
  SELECT 
    u.name,
    COUNT(CASE WHEN o.status = 'completed' THEN 1 END) as completed_orders,
    ARRAY_AGG(DISTINCT i.product_id) as purchased_products
  FROM (SELECT UNNEST(data.users) AS u FROM json_data AS data) AS users
  LEFT JOIN UNNEST(u.orders) AS o
  LEFT JOIN UNNEST(o.items) AS i
  GROUP BY u.name

- Nested JSON aggregations with window functions:
  WITH user_orders AS (
    SELECT 
      u.name,
      o.date as order_date,
      ROW_NUMBER() OVER (PARTITION BY u.id ORDER BY o.date) as order_num
    FROM (SELECT UNNEST(data.users) AS u FROM json_data AS data) AS users
    LEFT JOIN UNNEST(u.orders) AS o
  )
  SELECT 
    name,
    order_date,
    order_num,
    DATE_DIFF('day', LAG(order_date) OVER (PARTITION BY name ORDER BY order_date), order_date) as days_since_last_order
  FROM user_orders

Remember to:
- the JSON data is very large and complex, so you must use SQL to understand the schema first
- Leverage DuckDB's full SQL capabilities including window functions, CTEs, and subqueries
- Write queries that handle NULL values and edge cases appropriately
- Use appropriate array functions (UNNEST, ARRAY_AGG, ARRAY_LENGTH) for nested data
- Consider performance implications when dealing with large nested structures
"""

# The Schema Agent
large_json_agent = create_react_agent(
    model=model,
    tools=[query_json_tool],
    name="large_json_analyst",
    prompt=system_prompt
)




# Supervisor for JSON Data Handling
#json_supervisor = create_supervisor(
#    [json_query_agent] + section_agents,
#    model=model,
#    prompt="You are responsible for managing JSON section experts. Route data-related queries properly.",
#    output_mode="full_history"
#).compile(name="json_supervisor")

# === Final Supervisor (Manages Schema & JSON Supervisors) ===

#final_supervisor = create_supervisor(
#    [schema_supervisor, json_supervisor],
#    model=model,
#    prompt="You are the top-level supervisor managing both JSON schema and JSON data experts. "
#           "Route queries appropriately based on whether they concern schema or actual data.",
#    output_mode="full_history"
#)

# === Running the Multi-Agent System ===

#final_supervisor_graph = final_supervisor.compile()

# Example Query
#user_input = """JSON Data path: data/test_data.json
#Schema path: data/test_data_schema.json

#How many orders are there?"""

#query_input = {
#    "messages": [
#        {
#            "role": "user",
#            "content": user_input
#        }
#    ]
#}

#result = app.invoke(query_input)
#print(result)
