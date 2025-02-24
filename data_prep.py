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

def create_table(conn, table_name, schema):
    """ Creates a table for a top-level key in the JSON document. """
    columns = ", ".join([f'"{col}" TEXT' for col in schema])
    query = f'CREATE TABLE IF NOT EXISTS "{table_name}" (record_id UUID PRIMARY KEY, {columns})'
    conn.execute(query)

def insert_data(conn, table_name, data):
    """ Inserts data into the corresponding table. """
    if isinstance(data, list):  # Handle lists by inserting each item separately
        for item in data:
            insert_data(conn, table_name, item)
    elif isinstance(data, dict):
        columns = ", ".join([f'"{k}"' for k in data.keys()])
        placeholders = ", ".join(["?" for _ in data])
        query = f'INSERT INTO "{table_name}" (record_id, {columns}) VALUES (?, {placeholders})'
        values = [uuid.uuid4()] + list(map(str, data.values()))
        conn.execute(query, values)

def index_json(db_name, json_data, collection_name, embedder, qdrant_client=None):
    """ Parses a JSON document and indexes it into DuckDB and Qdrant. """
    conn = create_db(db_name)
    points = []
    
    for key, value in json_data.items():
        if isinstance(value, list) and all(isinstance(i, dict) for i in value):
            schema = set()
            for item in value:
                schema.update(item.keys())
            create_table(conn, key, schema)
            insert_data(conn, key, value)
        elif isinstance(value, dict):
            create_table(conn, key, value.keys())
            insert_data(conn, key, value)
        else:
            create_table(conn, key, ["value"])
            insert_data(conn, key, {"value": value})
        
        # Generate embeddings and store in Qdrant
        #value_str = str(value)
        #embedding = embedder.embed_query(value_str)  # Using LangChain's embed_query method
        #points.append(PointStruct(id=str(uuid.uuid4()), vector=embedding, payload={"json_path": key, "json_value": value_str}))
    
    #if points:
    #    qdrant_client.upsert(collection_name=collection_name, points=points)
    
    conn.commit()
    conn.close()
    print(f"Indexed JSON into {db_name}.duckdb and stored embeddings in Qdrant")
