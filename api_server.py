#!/usr/bin/env python3
"""
Simple FastAPI REST API server
"""

import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/hello", response_model=HelloResponse)
async def say_hello(request: HelloRequest):
    """Say hello to a user"""
    return HelloResponse(message=f"Hello {request.username.capitalize()}!")

def main():
    """Main function to run the server"""
    print(f"Starting FastAPI server on {HOST}:{PORT}...")
    uvicorn.run(app, host=HOST, port=PORT)

if __name__ == "__main__":
    main()
