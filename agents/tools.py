from smolagents import tool, Tool
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
import duckdb

# Initialize Qdrant client and embedder
#qdrant_client = QdrantClient(url="http://localhost:6333")  # Using in-memory storage
embedder = OpenAIEmbeddings(model="text-embedding-3-small")


@tool
def query_duckdb(db_name: str, query: str) -> str:
    """
    Execute a SQL query on a DuckDB database.
    
    Args:
        db_name: Name of the DuckDB database to query
        query: SQL query to execute
        
    Returns:
        str: A formatted string containing the query results in a tabular format,
             or "No results found" if the query returns no data
    """
    conn = duckdb.connect(f"{db_name}.duckdb")
    cursor = conn.execute(query)
    result = cursor.fetchall()
    
    if not result:
        conn.close()
        return "No results found"
    
    # Get column names from the cursor before closing the connection
    columns = [desc[0] for desc in cursor.description]
    
    # Now we can safely close the connection
    conn.close()
    
    # Format as table
    table = "| " + " | ".join(columns) + " |\n"
    table += "|" + "|".join(["-"*len(col) for col in columns]) + "|\n"
    
    for row in result:
        table += "| " + " | ".join(str(val) for val in row) + " |\n"
        
    return table

@tool
def semantic_search(db_name: str, query_text: str, top_k: int = 3) -> str:
    """
    Perform semantic search using Qdrant.
    
    Args:
        db_name: Name of the Qdrant collection to search
        query_text: Text to search for semantically similar matches
        top_k: Number of results to return (default: 3)
        
    Returns:
        str: A formatted string containing the search results with JSON paths and values,
             or "No results found" if no matches are found
    """
    embedding = embedder.embed_query(query_text)
    search_results = qdrant_client.search(
        collection_name=db_name,
        query_vector=embedding,
        limit=top_k
    )
    
    if not search_results:
        return "No results found"
        
    output = "Search Results:\n\n"
    for i, res in enumerate(search_results, 1):
        output += f"Result {i}:\n"
        output += f"JSON Path: {res.payload['json_path']}\n"
        output += f"Value: {res.payload['json_value']}\n\n"
        
    return output
