from pydantic import BaseModel
from typing import Union, Any
import json

from fastapi import FastAPI

from agents.duckdb_json_agent import large_json_agent, json_agents, db_connections
from fastapi import HTTPException


app = FastAPI()



class UserInput(BaseModel):
    user_input: str
    db_name: str


class MessageContent(BaseModel):
    role: str
    content: str


class Messages(BaseModel):
    messages: list[Any]


class QueryOutput(BaseModel):
    message_history: Messages
    final_answer: Any



@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/databases")
def list_databases():
    """List all available database instances."""
    return {"databases": list(db_connections.keys())}

@app.post("/query_json", response_model=QueryOutput)
async def query_json(user_input: UserInput):
    """Query JSON using the default database instance (llamacloud)."""
    USER_INPUT_TEMPLATE = f"""
Question: {user_input.user_input}
"""

    input = {
        "messages": [
            {
                "role": "user",
                "content": USER_INPUT_TEMPLATE
            }
        ]
    }

    def print_stream(stream):
        stream_output = []
        for s in stream:
            message = s["messages"][-1]
            # Print message based on its type
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()
            stream_output.append(Messages(messages=[message]))
        
        # Get the final message
        final_message = stream_output[-1].messages[-1]
        
        # Combine all messages into message history
        all_messages = []
        for output in stream_output:
            all_messages.extend(output.messages)
        
        return QueryOutput(
            message_history=Messages(messages=all_messages),
            final_answer=final_message
        )

    stream = large_json_agent.stream(input, stream_mode="values")
    return print_stream(stream)
    #return await large_json_agent.ainvoke(query_input)


@app.post("/query_named_json", response_model=QueryOutput)
async def query_named_json(user_input: UserInput):
    """Query JSON using a named database instance."""
    if user_input.db_name not in json_agents:
        raise HTTPException(status_code=404, detail=f"Database '{user_input.db_name}' not found. Available databases: {list(db_connections.keys())}")
    
    agent = json_agents[user_input.db_name]
    
    USER_INPUT_TEMPLATE = f"""
Question: {user_input.user_input}
"""

    input = {
        "messages": [
            {
                "role": "user",
                "content": USER_INPUT_TEMPLATE
            }
        ]
    }

    def print_stream(stream):
        stream_output = []
        for s in stream:
            message = s["messages"][-1]
            # Print message based on its type
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()
            stream_output.append(Messages(messages=[message]))
        
        # Get the final message
        final_message = stream_output[-1].messages[-1]
        
        # Combine all messages into message history
        all_messages = []
        for output in stream_output:
            all_messages.extend(output.messages)
        
        return QueryOutput(
            message_history=Messages(messages=all_messages),
            final_answer=final_message
        )

    stream = agent.stream(input, stream_mode="values")
    return print_stream(stream)
