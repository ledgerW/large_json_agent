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
from typing import Annotated, Dict
from pathlib import Path

# Initialize model
model = ChatOpenAI(model="gpt-4o")

# Global storage
db_connections: Dict[str, duckdb.DuckDBPyConnection] = {}
initialized_data: Dict[str, Dict] = {}  # Store loaded JSON data
db_tables: Dict[str, list] = {}  # Store table names for each database

def initialize_databases():
    """Initialize databases for all JSON files in the data directory."""
    data_dir = Path("data")
    if not data_dir.exists():
        raise Exception("Data directory not found")
    
    for json_file in data_dir.glob("*.json"):
        db_name = json_file.stem  # Use filename without extension as db name
        
        # Skip if already initialized
        if db_name in db_connections:
            continue
            
        try:
            # Create database connection with unique file per database
            db_dir = Path('/tmp/duckdb_dbs')
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / f'{db_name}.db'
            conn = duckdb.connect(str(db_path), read_only=False)
            db_connections[db_name] = conn
            
            # Load JSON data
            with open(json_file, 'rb') as f:
                data = orjson.loads(f.read())
            initialized_data[db_name] = data
            
            # Initialize tables list for this database
            db_tables[db_name] = []
            
            # Load tables
            for section, section_data in data.items():
                if isinstance(section_data, dict):
                    section_data = [section_data]
                elif not isinstance(section_data, list):
                    continue
                    
                try:
                    df = json_normalize(section_data, sep='.')
                    conn.register(section, df)
                    db_tables[db_name].append(section)
                    print(f"Initialized table {section} in database {db_name}")
                except Exception as e:
                    print(f"Error processing section {section} in {db_name}: {str(e)}")
                    continue
                    
            print(f"Successfully initialized database: {db_name}")
            
        except Exception as e:
            print(f"Error initializing database {db_name}: {str(e)}")
            continue

def get_named_db(db_name: str) -> duckdb.DuckDBPyConnection:
    """Get an initialized DuckDB connection."""
    if db_name not in db_connections:
        raise ValueError(f"Database {db_name} not initialized")
    return db_connections[db_name]

def query_json(query: str, db_name: str) -> str:
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
        # Get initialized database connection
        con = get_named_db(db_name)
        
        # Execute query
        result = con.execute(query).df()
        
        # Convert result to string representation
        if len(result) == 0:
            return "No results found"
        return result.to_string()
        
    except Exception as e:
        return f"Error querying JSON: {str(e)}"


def create_query_json_tool(db_name: str):
    """Create a query_json_tool for a specific database."""
    @tool("query_json_data")
    def query_json_tool(
        query: Annotated[str, """The DuckDB SQL query to execute against the available tables. Uses standard SQL syntax."""]
    ) -> Annotated[str, "Query results as a string"]:
        """
        Use this tool to query the JSON data file using SQL via DuckDB.
        
        The JSON data is automatically loaded with each top-level section as a separate table.
        Nested objects and arrays within each section are expanded into columns using dot notation.
        Arrays are properly maintained as lists where appropriate.
        
        Available tables in database '{db_name}':
        {db_tables.get(db_name, [])}
        
        You can query across multiple tables using standard SQL JOIN operations.
        """
        return query_json(query, db_name)
    
    return query_json_tool


def get_system_prompt(db_name: str) -> str:
    """Generate system prompt for a specific database."""
    tables = db_tables.get(db_name, [])
    sections_str = "\n".join([f"- {table}" for table in tables])
    
    return f"""
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


Database: {db_name}

Available tables in this database:
{sections_str}
"""

def get_json_agent(db_name: str = "default"):
    """Create a JSON agent for a specific named database."""
    query_tool = create_query_json_tool(db_name)
    prompt = get_system_prompt(db_name)
    
    return create_react_agent(
        model=model,
        tools=[query_tool],
        name=f"large_json_analyst_{db_name}",
        prompt=prompt
    )

# Initialize databases during module import
initialize_databases()

# Create agents for each database
json_agents = {db_name: get_json_agent(db_name) for db_name in db_connections}

# Create default agent for backward compatibility (using llamacloud database)
large_json_agent = json_agents.get("llamacloud")
