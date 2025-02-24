from pydantic import BaseModel
from typing import Union, Any
import json

from fastapi import FastAPI

from agents.duckdb_json_agent import large_json_agent


app = FastAPI()



class UserInput(BaseModel):
    user_input: str


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


@app.post("/query_json", response_model=QueryOutput)
async def query_json(user_input: UserInput):
    # Get available sections from the schema
    #json_file_path = "data/llama-cloud-history.json"

    #with open(json_file_path, "r") as f:
    #    json_data = json.load(f)
    #json_sections = list(json_data.keys())
    #json_sections_str = "\n".join([f"- {section}" for section in json_sections])


    USER_INPUT_TEMPLATE = f"""
    JSON Data file path: data/llama-cloud-history.json
    
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
