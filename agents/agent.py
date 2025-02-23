import os
from dotenv import load_dotenv
load_dotenv()

import json
from spyql.query import Query
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
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




def query_json(query: str, json_file_path: str) -> str:
    """
    Query a JSON file using SpyQL.
    
    This function loads the JSON file and runs the provided SpyQL query against it.
    SpyQL's powerful query syntax can directly access nested data without needing
    separate path navigation.
    
    Args:
        query: SpyQL query to execute on the JSON data. Can use dot notation 
              to access nested fields directly in the query.
        json_file_path: Path to the JSON file to query
        
    Returns:
        The query results as a string, or an error message if the query fails
    """
    #try:
    # Create SpyQL query with the JSON file loaded as a variable
    spyql_query = Query(
        query,
        json_obj_files={"data": json_file_path}  # Load JSON file as 'data' variable
    )
    result = spyql_query()  # Execute query
    return str(result)
    #except Exception as e:
    #    return f"Error querying JSON: {str(e)}"


@tool("query_json_data")
def query_json_tool(
    query: Annotated[str, """The SPyQL query to execute. Supports SPyQL syntax ONLY."""],
    json_file_path: Annotated[str, "Path to the JSON file to query"]
) -> Annotated[str, "Query results as a string"]:
    """
    Use this tool to query the JSON data file using SPyQL's query syntax.
    
    SPyQL provides querying capabilities with SQL syntax plus Python expressions.
    The JSON file is loaded into a 'data' variable that can be queried.
    """
    return query_json(query, json_file_path)





# Get available sections from the schema
schema_path = "data/test_data_schema.json"
json_file_path = "data/test_data.json"

with open(schema_path, "r") as f:
    schema_data = json.load(f)
json_sections = list(schema_data.get("properties", {}).keys())



system_prompt = """
You are an expert JSON data analyst that uses SpyQL to query JSON data.
You are given a JSON schema and a JSON data file, with which you must answer questions
about the data.

You must use SpyQL to query the data, and you must use the schema to understand how to
query the data.

The data and the schema are both very large and complex, so you must use the tools, and
you must use the schema to understand how to query the data.

SPyQL Query syntax:
    [ IMPORT python_module [ AS identifier ] [, ...] ]
    SELECT [ DISTINCT | PARTIALS ]
        [ * | python_expression [ AS output_column_name ] [, ...] ]
        [ FROM csv | spy | text | python_expression | json [ EXPLODE path ] ]
        [ WHERE python_expression ]
        [ GROUP BY output_column_number | python_expression  [, ...] ]
        [ ORDER BY output_column_number | python_expression
            [ ASC | DESC ] [ NULLS { FIRST | LAST } ] [, ...] ]
        [ LIMIT row_count ]
        [ OFFSET num_rows_to_skip ]
        [ TO csv | json | spy | sql | pretty | plot ]
    
    Example queries:
      Basic query:
        SELECT .my_key
        FROM data
      Nested query:
        SELECT row.my_key
        FROM data
      Dictionary query:
        SELECT row['my_key']
        FROM data

      With WHERE clause (uses python expressions):
        SELECT .name
        FROM [
            {"name": "Alice", "age": 20, "salary": 30.0},
            {"name": "Bob", "age": 30, "salary": 12.0},
            {"name": "Charles", "age": 40, "salary": 6.0},
            {"name": "Daniel", "age": 43, "salary": 0.40},
        ]
        WHERE .age > 30 and .salary < 10.0 or .name.startswith('D')


      Group by and order by:
        SELECT .name, .score
        FROM json
        GROUP BY .department
        ORDER BY .score DESC NULLS LAST
        LIMIT 5
        OFFSET 1


      Group by and order by with aggregate function:
        SELECT .player_name, sum_agg(.score) AS total_score
        FROM json
        GROUP BY 1
        ORDER BY 1


      Explode and JSON output:
        SELECT .name, .departments
        FROM [
            {"name": "Alice", "departments": [1,4]},
            {"name": "Bob", "departments": [2]},
            {"name": "Charles", "departments": []}
        ]
        EXPLODE .departments
        TO json

        Output:
        {"name": "Alice", "departments": 1}
        {"name": "Alice", "departments": 4}
        {"name": "Bob", "departments": 2}


      Pretty Output:
        SELECT .id, .name
        FROM json
        TO pretty

        Output:
        id  name
        -----  -----------
        23635  Jerry Green
        23636  John Wayne 
"""

# The Schema Agent
large_json_agent = create_react_agent(
    model=model,
    tools=[view_available_sections, view_schema_section, query_json_tool],
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
