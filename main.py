from pydantic import BaseModel
from typing import Union, Any, Dict, List
import json
import os
import duckdb
import glob

from fastapi import FastAPI

from agents.smolagent import task_agent
from fastapi import HTTPException
from data_prep import index_json
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings

# Initialize Qdrant client and embedder
#qdrant_client = QdrantClient(url="http://localhost:6333")  # Using in-memory storage
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# Load JSON files at startup
def load_json_files():
    data_dir = "data"
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            db_name = os.path.splitext(filename)[0]
            file_path = os.path.join(data_dir, filename)
            # Create Qdrant collection for this file
            #qdrant_client.recreate_collection(
            #    collection_name=db_name,
            #    vectors_config={"size": 1536, "distance": "Cosine"}  # OpenAI embeddings are 1536 dimensions
            # )
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                index_json(db_name, json_data, db_name, embedder)
                print(f"Loaded and indexed {filename} into collection {db_name}")

# Initialize FastAPI app
app = FastAPI()

# Load JSON files when the application starts
@app.on_event("startup")
async def startup_event():
    load_json_files()



class TaskInput(BaseModel):
    task: str
    db_name: str

class TaskOutput(BaseModel):
    output: Any


@app.get("/")
def read_root():
    return {"Hello": "World"}


def get_database_info() -> Dict[str, List[str]]:
    """
    Scan for DuckDB databases and return their table information.
    Returns a dictionary with database names as keys and lists of table names as values.
    """
    database_info = {}
    
    # Find all .duckdb files in the current directory and subdirectories
    duckdb_files = glob.glob("**/*.duckdb", recursive=True)
    
    for db_path in duckdb_files:
        try:
            # Connect to the database
            conn = duckdb.connect(db_path, read_only=True)
            
            # Get all tables in the database
            tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
            
            # Store the database name and its tables
            db_name = os.path.basename(db_path)
            database_info[db_name] = [table[0] for table in tables]
            
            # Close the connection
            conn.close()
        except Exception as e:
            print(f"Error accessing database {db_path}: {str(e)}")
            continue
    
    return database_info

@app.get("/databases")
def list_databases():
    """List all available database instances and their tables."""
    return {"databases": get_database_info()}


@app.post("/task_agent", response_model=TaskOutput)
def run_task_agent(input: TaskInput) -> TaskOutput:
    """Execute a task using the task agent that combines SQL and semantic search capabilities"""
    #try:
    task_template = f"""
    Your task is to answer the question as best you can.
    All of your queries will be executed on the database below.
    
    DB Name: {input.db_name}.
    
    Question: {input.task}
    """
    result = task_agent.run(task_template)
    return TaskOutput(output=result)
    #except Exception as e:
    #    raise HTTPException(status_code=500, detail=str(e))
