from smolagents import tool, Tool
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
import duckdb
import json

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
def get_hierarchical_data_info(db_name: str = None) -> str:
    """
    Retrieve and display information about the hierarchical structure of a database.
    This tool provides an overview of the database structure, table hierarchy, table statistics,
    and relationships between tables.
    
    Args:
        db_name: Name of the DuckDB database to analyze. If not provided, it will use the db_name from additional_args.
        
    Returns:
        str: A formatted string containing detailed information about the hierarchical data structure
    """
    try:
        conn = duckdb.connect(f"{db_name}.duckdb")
        
        # Check if this is a hierarchical database with schema_info
        try:
            conn.execute("SELECT * FROM schema_info LIMIT 1")
        except:
            conn.close()
            return f"Error: {db_name} does not appear to be a hierarchical database with schema_info table."
        
        output = []
        
        # Query 1: Get an overview of the data structure
        output.append("=== Data Overview ===")
        try:
            result = conn.execute("SELECT * FROM data_overview").fetchall()
            for row in result:
                for col in row:
                    output.append(str(col))
        except:
            output.append("No data overview available.")
        
        # Query 2: Show the table hierarchy
        output.append("\n=== Table Hierarchy ===")
        result = conn.execute("SELECT level, path, table_name, is_array, count FROM table_hierarchy ORDER BY path").fetchall()
        output.append("Level | Path | Table Name | Is Array | Count")
        output.append("-" * 70)
        for row in result:
            output.append(f"{row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]}")
        
        # Query 3: Show table statistics
        output.append("\n=== Table Statistics ===")
        result = conn.execute("SELECT table_name, is_array, count AS expected_count, description FROM schema_info ORDER BY table_name").fetchall()
        output.append("Table Name | Is Array | Expected Count | Description")
        output.append("-" * 80)
        for row in result:
            output.append(f"{row[0]} | {row[1]} | {row[2]} | {row[3]}")
        
        # Query 4: Find the root tables (tables with no parent)
        output.append("\n=== Root Tables ===")
        root_tables = conn.execute("""
            SELECT table_name, count, description 
            FROM schema_info 
            WHERE parent_table IS NULL
            ORDER BY table_name
        """).fetchall()
        
        if not root_tables:
            output.append("No root tables found.")
        else:
            output.append("Table Name | Count | Description")
            output.append("-" * 60)
            for row in root_tables:
                output.append(f"{row[0]} | {row[1]} | {row[2]}")
                
                # For each root table, show record count comparison
                try:
                    table_name = row[0]
                    raw_count = conn.execute(f"SELECT COUNT(*) FROM \"{table_name}\"").fetchall()
                    schema_count = row[1]  # Count from schema_info
                    
                    output.append(f"\nRecord counts for {table_name}:")
                    output.append(f"Raw count of records in the {table_name} table: {raw_count[0][0]}")
                    output.append(f"Expected number according to schema_info: {schema_count}")
                    if raw_count[0][0] != schema_count:
                        output.append("The difference is due to the hierarchical structure of the data.")
                except:
                    output.append(f"Could not get record count for {table_name}.")
        
        # Query 5: Show key relationships between tables
        output.append("\n=== Table Relationships ===")
        result = conn.execute("""
            SELECT 
                r.parent_table, 
                r.child_table, 
                COUNT(*) as relationship_count,
                s.description
            FROM record_relationships r
            JOIN schema_info s ON r.child_table = s.table_name
            GROUP BY r.parent_table, r.child_table, s.description
            ORDER BY relationship_count DESC
            LIMIT 20
        """).fetchall()
        
        if not result:
            output.append("No relationships found.")
        else:
            output.append("Parent Table | Child Table | Relationship Count | Description")
            output.append("-" * 80)
            for row in result:
                output.append(f"{row[0]} | {row[1]} | {row[2]} | {row[3]}")
        
        conn.close()
        return "\n".join(output)
    
    except Exception as e:
        return f"Error analyzing hierarchical data: {str(e)}"

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
