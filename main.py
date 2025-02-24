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
                print(f"Loaded and indexed {filename} into collection {db_name} with hierarchical structure preserved")

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


def get_database_info() -> Dict[str, Dict[str, Any]]:
    """
    Scan for DuckDB databases and return their table information and structure.
    Returns a dictionary with database names as keys and detailed information about each database.
    """
    database_info = {}
    
    # Find all .duckdb files in the current directory and subdirectories
    duckdb_files = glob.glob("**/*.duckdb", recursive=True)
    
    for db_path in duckdb_files:
        try:
            # Connect to the database
            conn = duckdb.connect(db_path, read_only=True)
            db_name = os.path.basename(db_path)
            
            # Check if this is a hierarchical database with schema_info
            has_schema_info = False
            try:
                conn.execute("SELECT * FROM schema_info LIMIT 1")
                has_schema_info = True
            except:
                pass
            
            if has_schema_info:
                # Get hierarchical structure information
                tables = conn.execute("SELECT table_name, parent_table, description, is_array, count FROM schema_info").fetchall()
                table_stats = conn.execute("SELECT table_name, COUNT(*) as record_count FROM schema_info JOIN (SELECT name FROM sqlite_master WHERE type='table' AND name NOT IN ('schema_info', 'record_relationships')) t ON schema_info.table_name = t.name GROUP BY table_name").fetchall()
                
                # Get overview if available
                overview = None
                try:
                    overview = conn.execute("SELECT * FROM data_overview").fetchall()
                except:
                    pass
                
                database_info[db_name] = {
                    "hierarchical": True,
                    "tables": [{"name": t[0], "parent": t[1], "description": t[2], "is_array": t[3], "count": t[4]} for t in tables],
                    "stats": {t[0]: t[1] for t in table_stats},
                    "overview": overview
                }
            else:
                # Get basic table information for non-hierarchical databases
                tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
                database_info[db_name] = {
                    "hierarchical": False,
                    "tables": [{"name": t[0]} for t in tables]
                }
            
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
    # Get database structure information
    db_info = get_database_info().get(f"{input.db_name}.duckdb", {})
    
    # Create a context-rich prompt based on database structure
    if db_info.get("hierarchical", False):
        # For hierarchical databases, provide rich context about the structure
        conn = duckdb.connect(f"{input.db_name}.duckdb", read_only=True)
        
        # Get table hierarchy
        hierarchy = conn.execute("SELECT level, path, table_name, is_array, count FROM table_hierarchy ORDER BY path").fetchall()
        hierarchy_str = "\n".join([f"- Level {row[0]}: {row[1]} (Array: {row[3]}, Count: {row[4]})" for row in hierarchy])
        
        # Get table statistics
        stats = conn.execute("SELECT table_name, expected_count FROM table_statistics ORDER BY expected_count DESC").fetchall()
        stats_str = "\n".join([f"- {row[0]}: {row[1]} records" for row in stats])
        
        # Get overview
        overview = "No overview available"
        try:
            overview_result = conn.execute("SELECT * FROM data_overview").fetchall()
            if overview_result:
                overview = "\n".join([str(col) for col in overview_result[0]])
        except:
            pass
        
        conn.close()
        
        task_template = f"""
        Your task is to answer the question as best you can.
        All of your queries will be executed on the database below.
        
        DB Name: {input.db_name}
        
        Database Overview:
        {overview}
        
        Table Hierarchy:
        {hierarchy_str}
        
        Table Statistics:
        {stats_str}
        
        IMPORTANT: This database contains hierarchical JSON data. 
        The schema_info table shows that the 'jobs' table contains {db_info.get('tables', [{}])[0].get('count', '?')} original job records.
        When counting records in the jobs table, you may see more records due to the hierarchical structure.
        To get accurate counts, refer to the schema_info table or use the table_hierarchy and table_statistics views.
        
        Question: {input.task}
        """
    else:
        # For regular databases, provide basic context
        tables = [t.get("name") for t in db_info.get("tables", [])]
        tables_str = ", ".join(tables) if tables else "No tables found"
        
        task_template = f"""
        Your task is to answer the question as best you can.
        All of your queries will be executed on the database below.
        
        DB Name: {input.db_name}
        Available tables: {tables_str}
        
        Question: {input.task}
        """
    
    # Pass the db_name as additional_args to the agent
    result = task_agent.run(task_template, additional_args={"db_name": input.db_name})
    return TaskOutput(output=result)
    #except Exception as e:
    #    raise HTTPException(status_code=500, detail=str(e))
