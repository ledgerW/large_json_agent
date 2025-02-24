import json
import duckdb
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from langchain_openai import OpenAIEmbeddings

def create_db(db_name):
    """ Creates a new DuckDB database for each JSON document. """
    conn = duckdb.connect(f"{db_name}.duckdb")
    return conn

def create_schema_tables(conn):
    """
    Creates tables for storing schema information and hierarchical relationships.
    This helps the LLM understand the structure of the data.
    """
    # Create a table for storing schema information
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_info (
            table_name TEXT PRIMARY KEY,
            parent_table TEXT,
            description TEXT,
            is_array BOOLEAN,
            count INTEGER
        )
    """)
    
    # Create a table for storing relationships between records
    conn.execute("""
        CREATE TABLE IF NOT EXISTS record_relationships (
            child_id UUID,
            parent_id UUID,
            child_table TEXT,
            parent_table TEXT,
            relationship_type TEXT
        )
    """)

def create_table(conn, table_name, schema, parent_table=None, is_array=False, count=None, description=None):
    """ 
    Creates a table for a top-level key in the JSON document and records schema information.
    """
    # Create the data table
    columns = ", ".join([f'"{col}" TEXT' for col in schema])
    query = f'CREATE TABLE IF NOT EXISTS "{table_name}" (record_id UUID PRIMARY KEY, {columns})'
    conn.execute(query)
    
    # Record schema information
    conn.execute("""
        INSERT OR REPLACE INTO schema_info 
        (table_name, parent_table, description, is_array, count) 
        VALUES (?, ?, ?, ?, ?)
    """, (table_name, parent_table, description, is_array, count))

def insert_data(conn, table_name, data, parent_id=None, parent_table=None):
    """ 
    Inserts data into the corresponding table while maintaining hierarchical relationships.
    """
    if isinstance(data, list):
        # For arrays, insert each item and maintain the relationship to the parent
        for i, item in enumerate(data):
            if isinstance(item, dict):
                # Generate a record ID for this item
                record_id = uuid.uuid4()
                
                # Insert the item
                columns = ", ".join([f'"{k}"' for k in item.keys()])
                placeholders = ", ".join(["?" for _ in item])
                query = f'INSERT INTO "{table_name}" (record_id, {columns}) VALUES (?, {placeholders})'
                values = [record_id] + list(map(str, item.values()))
                conn.execute(query, values)
                
                # Record the relationship to the parent if applicable
                if parent_id:
                    conn.execute("""
                        INSERT INTO record_relationships 
                        (child_id, parent_id, child_table, parent_table, relationship_type) 
                        VALUES (?, ?, ?, ?, ?)
                    """, (record_id, parent_id, table_name, parent_table, f"array_item[{i}]"))
                
                # Process nested objects
                for key, value in item.items():
                    if isinstance(value, (dict, list)) and value:
                        nested_table_name = f"{table_name}_{key}"
                        process_nested_data(conn, nested_table_name, value, record_id, table_name)
    
    elif isinstance(data, dict):
        # For objects, insert the record and process nested fields
        record_id = uuid.uuid4()
        
        # Insert the record
        columns = ", ".join([f'"{k}"' for k in data.keys()])
        placeholders = ", ".join(["?" for _ in data])
        query = f'INSERT INTO "{table_name}" (record_id, {columns}) VALUES (?, {placeholders})'
        values = [record_id] + list(map(str, data.values()))
        conn.execute(query, values)
        
        # Record the relationship to the parent if applicable
        if parent_id:
            conn.execute("""
                INSERT INTO record_relationships 
                (child_id, parent_id, child_table, parent_table, relationship_type) 
                VALUES (?, ?, ?, ?, ?)
            """, (record_id, parent_id, table_name, parent_table, "object_field"))
        
        # Process nested objects
        for key, value in data.items():
            if isinstance(value, (dict, list)) and value:
                nested_table_name = f"{table_name}_{key}"
                process_nested_data(conn, nested_table_name, value, record_id, table_name)

def process_nested_data(conn, table_name, data, parent_id, parent_table):
    """
    Process nested data structures (objects or arrays) and maintain relationships.
    """
    if isinstance(data, list) and data and all(isinstance(i, dict) for i in data):
        # For arrays of objects, create a table with all possible fields
        schema = set()
        for item in data:
            schema.update(item.keys())
        
        create_table(
            conn, 
            table_name, 
            schema, 
            parent_table=parent_table, 
            is_array=True, 
            count=len(data),
            description=f"Array of {len(data)} items from {parent_table}"
        )
        
        # Insert the array items
        insert_data(conn, table_name, data, parent_id, parent_table)
    
    elif isinstance(data, dict):
        # For objects, create a table with its fields
        create_table(
            conn, 
            table_name, 
            data.keys(), 
            parent_table=parent_table,
            is_array=False,
            description=f"Object field from {parent_table}"
        )
        
        # Insert the object
        insert_data(conn, table_name, data, parent_id, parent_table)

def create_metadata_views(conn):
    """
    Create views that help the LLM understand the data structure.
    """
    # Create a view that shows the table hierarchy
    conn.execute("""
        CREATE OR REPLACE VIEW table_hierarchy AS
        WITH RECURSIVE hierarchy AS (
            SELECT 
                table_name, 
                parent_table, 
                description,
                is_array,
                count,
                0 AS level,
                table_name AS path
            FROM schema_info
            WHERE parent_table IS NULL
            
            UNION ALL
            
            SELECT 
                s.table_name, 
                s.parent_table, 
                s.description,
                s.is_array,
                s.count,
                h.level + 1,
                h.path || '.' || s.table_name
            FROM schema_info s
            JOIN hierarchy h ON s.parent_table = h.table_name
        )
        SELECT 
            level,
            path,
            table_name, 
            parent_table, 
            description,
            is_array,
            count
        FROM hierarchy
        ORDER BY path
    """)
    
    # Create a very simple view that shows table statistics
    conn.execute("""
        CREATE OR REPLACE VIEW table_statistics AS
        SELECT 
            table_name,
            is_array,
            count AS expected_count,
            description
        FROM schema_info
        ORDER BY table_name
    """)

def index_json(db_name, json_data, collection_name, embedder=None, qdrant_client=None):
    """ 
    Parses a JSON document and indexes it into DuckDB and Qdrant while maintaining
    hierarchical relationships and schema information.
    """
    conn = create_db(db_name)
    points = []  # Initialize points list for embeddings
    
    # Create schema tables
    create_schema_tables(conn)
    
    # Process top-level keys
    for key, value in json_data.items():
        if isinstance(value, list) and all(isinstance(i, dict) for i in value):
            # For arrays of objects (like the "jobs" array)
            schema = set()
            for item in value:
                schema.update(item.keys())
            
            # Create a table for this array
            create_table(
                conn, 
                key, 
                schema, 
                is_array=True, 
                count=len(value),
                description=f"Top-level array containing {len(value)} items"
            )
            
            # Insert the array items
            insert_data(conn, key, value)
            
            # Generate embeddings for the array (optional)
            if qdrant_client and embedder:
                for i, item in enumerate(value):
                    item_str = json.dumps(item)
                    embedding = embedder.embed_query(item_str)
                    points.append(
                        PointStruct(
                            id=str(uuid.uuid4()), 
                            vector=embedding, 
                            payload={
                                "json_path": f"{key}[{i}]", 
                                "json_value": item_str,
                                "table": key
                            }
                        )
                    )
        
        elif isinstance(value, dict):
            # For objects
            create_table(
                conn, 
                key, 
                value.keys(),
                description=f"Top-level object"
            )
            insert_data(conn, key, value)
            
            # Generate embeddings for the object (optional)
            if qdrant_client and embedder:
                value_str = json.dumps(value)
                embedding = embedder.embed_query(value_str)
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()), 
                        vector=embedding, 
                        payload={
                            "json_path": key, 
                            "json_value": value_str,
                            "table": key
                        }
                    )
                )
        
        else:
            # For primitive values
            create_table(
                conn, 
                key, 
                ["value"],
                description=f"Top-level scalar value"
            )
            insert_data(conn, key, {"value": value})
    
    # Create metadata views to help the LLM understand the data structure
    create_metadata_views(conn)
    
    # Store embeddings in Qdrant if available
    if qdrant_client and points:
        qdrant_client.upsert(collection_name=collection_name, points=points)
    
    # Create a special view for the LLM to understand the data structure
    conn.execute("""
        CREATE OR REPLACE VIEW data_overview AS
        SELECT 
            'This database contains ' || COUNT(DISTINCT table_name) || ' tables representing a JSON document.' AS overview,
            (SELECT s.count FROM schema_info s WHERE s.table_name = 'jobs') || ' jobs are stored in the "jobs" table.' AS job_count,
            'Use the table_hierarchy and table_statistics views to understand the data structure.' AS hint,
            'IMPORTANT: The jobs table contains the original array items. Nested objects are stored in separate tables with relationships.' AS data_structure_hint
        FROM schema_info
    """)
    
    conn.commit()
    conn.close()
    print(f"Indexed JSON into {db_name}.duckdb with hierarchical structure preserved")
