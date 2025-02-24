from dotenv import load_dotenv
load_dotenv()

import os
#from phoenix.otel import register
#from openinference.instrumentation.smolagents import SmolagentsInstrumentor
import yaml
import pathlib

from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    LiteLLMModel
)

from agents.tools import query_duckdb, semantic_search, get_hierarchical_data_info

# configure the Phoenix tracer
#tracer_provider = register(
#  project_name=os.getenv("PHOENIX_PROJECT_NAME")
#)

#SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider)



# LLM
llm = LiteLLMModel(
    "openai/gpt-4o",
    temperature=0.1,
    max_tokens=2000,
    api_key=os.getenv("OPENAI_API_KEY")
)


# Tool Calling Agents
sql_query_agent = ToolCallingAgent(
    tools=[query_duckdb, get_hierarchical_data_info],
    model=llm,
    max_steps=10,
    name="sql_query_agent",
    description="This agent is used for structured queries on the DuckDB database and analyzing hierarchical data structures."
)

semantic_search_agent = ToolCallingAgent(
    tools=[semantic_search],
    model=llm,
    max_steps=10,
    name="semantic_search_agent",
    description="This agent is used for semantic search on the Qdrant collection."
)


# Manager CodeAgent
task_agent = CodeAgent(
    tools=[],
    model=llm,
    managed_agents=[sql_query_agent],
    additional_authorized_imports=["time", "numpy", "pandas"]
)
