from pydantic import BaseModel
from typing import Union
import json

from fastapi import FastAPI

from agents.duckdb_json_agent import large_json_agent


app = FastAPI()



class UserInput(BaseModel):
    user_input: str


class MessageContent(BaseModel):
    role: str
    content: str


class Message(BaseModel):
    messages: list[Union[MessageContent, tuple]]


class QueryOutput(BaseModel):
    message: Message



@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/query_json", response_model=QueryOutput)
async def query_json(user_input: UserInput):
    # Get available sections from the schema
    json_file_path = "data/test_data.json"

    with open(json_file_path, "r") as f:
        json_data = json.load(f)
    json_sections = list(json_data.keys())
    json_sections_str = "\n".join([f"- {section}" for section in json_sections])


    USER_INPUT_TEMPLATE = f"""
    JSON Data file path: data/test_data.json
    
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
            # Create a Message object with proper structure
            if isinstance(message, tuple):
                print(message)
                # Convert tuple to MessageContent
                message_content = MessageContent(role=message[0], content=message[1])
                stream_output.append(Message(messages=[message_content]))
            else:
                message.pretty_print()
                # Assuming message is already in correct format
                stream_output.append(Message(messages=[message]))
        return QueryOutput(messages=stream_output)

    stream = large_json_agent.stream(input, stream_mode="values")
    stream_messages = print_stream(stream)
    return stream_messages.message[-1]

    #return await large_json_agent.ainvoke(query_input)