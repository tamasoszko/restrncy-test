#!/usr/bin/env python3
"""
Simple FastAPI REST API server
"""

import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from crewai_flow import OutputExampleFlow
from openai_agents_workflow import RecommenderWorkflow


# import dotenv
# dotenv.load_dotenv()

HOST = os.getenv("API_SERVER_HOST", "0.0.0.0")
PORT = os.getenv("API_SERVER_PORT", 8000)

# Create FastAPI app
app = FastAPI(title="Simple API", version="1.0.0")



# Request model for hello endpoint
class HelloRequest(BaseModel):
    username: str

# Response model for hello endpoint
class HelloResponse(BaseModel):
    message: str

class ChatMessage(BaseModel):
    session_id: str | None = None
    message: str

class ChatResponse(ChatMessage):
    finished: bool
    history: list[dict[str, str]]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/hello", response_model=HelloResponse)
async def say_hello(request: HelloRequest):
    """Say hello to a user"""
    return HelloResponse(message=f"Hello {request.username.capitalize()}!")

@app.post("/chat/crewai_flow")
def chat_with_crewai_flow(request: ChatMessage):
    """Chat with CrewAI Flow"""
    flow = OutputExampleFlow()
    session_id, finished, message, history = flow.resume(id=request.session_id, user_input=request.message)
    return ChatResponse(message=message, session_id=session_id, finished=finished, history=history)

@app.post("/chat/restrncy")
async def chat_with_restaurant_finder(request: ChatMessage):
    """Chat with Restaurant Finder"""
    flow = RecommenderWorkflow(session_id=request.session_id)
    session_id, message = await flow.resume(user_input=request.message)
    # TODO: handle finished and history
    return ChatResponse(message=message, session_id=session_id, finished=False, history=[])



class ChatRequest(BaseModel):
    message: str

def main():
    """Main function to run the server"""
    print(f"Starting FastAPI server on {HOST}:{PORT}...")
    RecommenderWorkflow.setup()
    uvicorn.run(app, host=HOST, port=PORT)

if __name__ == "__main__":
    main()
