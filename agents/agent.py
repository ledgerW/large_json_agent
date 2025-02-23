import os
from dotenv import load_dotenv
load_dotenv()

import json
import spyql.query
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from typing import Annotated

# === Load and Process Large JSON and Schema in a Smart Way ===

def load_json_schema_section(schema_path: str, section_name: str) -> str:
    """Load only the relevant section of the large JSON schema."""
    with open(schema_path, "r") as f:
        schema = json.load(f)
    return json.dumps(schema.get("properties", {}).get(section_name, {}), indent=2)

def load_json_section(json_file_path: str, section_name: str) -> str:
    """Load only the relevant section of the large JSON file."""
    with open(json_file_path, "r") as f:
        data = json.load(f)
    return json.dumps(data.get(section_name, []), indent=2)

# Function to query JSON using SpyQL
def query_json_section(section_name: str, query: str, json_file_path: str) -> str:
    """Query a specific section of the large JSON file using SpyQL."""
    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)

        if section_name not in data:
            return f"Section '{section_name}' not found in the JSON document."

        section_data = data[section_name]  # Extract only the relevant section
        json_lines = "\n".join(json.dumps(record) for record in section_data)

        result = spyql.query.run_query(query, input=json_lines)
        return result
    except Exception as e:
        return f"Error querying JSON section: {str(e)}"

# Load model
model = ChatOpenAI(model="gpt-4o")

# === Schema Supervisor and Agents ===

# Schema Query Agent (Queries specific schema sections)
@tool
def get_schema_section(
    section_name: Annotated[str, "The name of the schema section to query"],
    schema_path: Annotated[str, "The path to the JSON schema file"]
) -> Annotated[str, "The section of the schema as a JSON string"]:
    """
    Use this tool to get a specific section of the JSON schema.
    """
    return load_json_schema_section(schema_path, section_name)


schema_query_agent = create_react_agent(
    model=model,
    tools=[get_schema_section],
    name="schema_query_agent",
    prompt="You query specific sections of the JSON schema when needed."
)

# Schema Agent (Lists available schema sections)
@tool   
def get_available_sections(
    schema_path: Annotated[str, "The path to the JSON schema file"]
) -> Annotated[str, "A list of available sections in the JSON schema"]:
    """
    Use this tool to get a list of available sections in the JSON schema.
    """
    with open(schema_path, "r") as f:
        schema = json.load(f)
    return f"Available sections in the JSON schema: {', '.join(schema.get('properties', {}).keys())}"


schema_agent = create_react_agent(
    model=model,
    tools=[get_available_sections],
    name="schema_expert",
    prompt="You are an expert in JSON schemas and help identify available sections."
)

# Supervisor for Schema Handling
schema_supervisor = create_supervisor(
    [schema_agent, schema_query_agent],
    model=model,
    prompt="You are responsible for managing JSON schema experts. Route schema-related queries properly."
).compile(name="schema_supervisor")

# === JSON Data Supervisor and Agents ===

# Create agents for each JSON section
def create_json_section_agent(section_name, json_file_path):
    """Create an agent for a specific JSON section."""
    
    @tool       
    def query_section(
        query: Annotated[str, "The SpyQL query to execute on the JSON section"],
        json_file_path: Annotated[str, "The path to the JSON file"]
    ) -> Annotated[str, "The result of the query"]:
        """
        Use this tool to query a specific section of the JSON file.
        """
        return query_json_section(section_name, query, json_file_path)

    return create_react_agent(
        model=model,
        tools=[query_section],
        name=f"{section_name}_expert",
        prompt=f"You are an expert JSON analyst that uses SpyQL to query the '{section_name}' section of the JSON file. You MUST use SpyQL to query the data."
    )

# Get available sections from the schema
schema_path = "data/test_data_schema.json"
json_file_path = "data/test_data.json"

with open(schema_path, "r") as f:
    schema_data = json.load(f)
json_sections = list(schema_data.get("properties", {}).keys())

# Create individual agents for each JSON section
section_agents = [create_json_section_agent(section, json_file_path) for section in json_sections]

# JSON Query Agent (Helps pick relevant sections)
@tool
def get_json_section(
    section_name: Annotated[str, "The name of the section to get"],
    json_file_path: Annotated[str, "The path to the JSON file"]
) -> Annotated[str, "The section of the JSON file"]:
    """
    Use this tool to get a specific section of the JSON file.
    """
    return load_json_section(json_file_path, section_name)

json_query_agent = create_react_agent(
    model=model,
    tools=[get_json_section],
    name="json_query_agent",
    prompt="You extract specific sections of the JSON data when needed."
)

# Supervisor for JSON Data Handling
json_supervisor = create_supervisor(
    [json_query_agent] + section_agents,
    model=model,
    prompt="You are responsible for managing JSON section experts. Route data-related queries properly."
).compile(name="json_supervisor")

# === Final Supervisor (Manages Schema & JSON Supervisors) ===

final_supervisor = create_supervisor(
    [schema_supervisor, json_supervisor],
    model=model,
    prompt="You are the top-level supervisor managing both JSON schema and JSON data experts. "
           "Route queries appropriately based on whether they concern schema or actual data."
)

# === Running the Multi-Agent System ===

app = final_supervisor.compile()

# Example Query
user_input = """JSON Data path: data/test_data.json
Schema path: data/test_data_schema.json

How many orders are there?"""

query_input = {
    "messages": [
        {
            "role": "user",
            "content": user_input
        }
    ]
}

result = app.invoke(query_input)
print(result)
