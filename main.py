from pydantic import BaseModel
from typing import Union

from fastapi import FastAPI

from agents.agent import large_json_agent


app = FastAPI()



class UserInput(BaseModel):
    user_input: str


class MessageContent(BaseModel):
    role: str
    content: str


class Message(BaseModel):
    messages: list[Union[MessageContent, tuple]]


class QueryOutput(BaseModel):
    messages: list[Message]



@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/query_json", response_model=QueryOutput)
async def query_json(user_input: UserInput):
    user_input = f"""
    JSON Data file path: data/test_data.json
    JSON Schema file path: data/test_data_schema.json
    
    Question: {user_input}
    """

    input = {
        "messages": [
            {
                "role": "user",
                "content": user_input
            }
        ]
    }

    def print_stream(stream):
        stream_output = []
        for s in stream:
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
                stream_output.append(message)
            else:
                message.pretty_print()
                stream_output.append(message)
        return stream_output

    stream = large_json_agent.stream(input, stream_mode="values")
    stream_messages = print_stream(stream)
    return stream_messages

    #return await large_json_agent.ainvoke(query_input)