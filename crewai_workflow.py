

import asyncio
import os
from typing import Any, Dict, List, Optional

from crewai import LLM, Agent, Task, Crew
from crewai.tools import tool
import dotenv
from pydantic import BaseModel
import openlit

dotenv.load_dotenv()

OPENAI_API_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openlit.init(disable_metrics=True, otlp_endpoint="http://127.0.0.1:4318")


class UserInputs(BaseModel):
    """What user inputs returns after asking for missing information."""
    topic: Optional[str] = None
    num_jokes: Optional[int] = None

class Presentation(BaseModel):
    """What presentation returns after presenting the result."""
    message: Optional[str] = None

class UserResponse(BaseModel):
    """What user response returns after presenting the result."""
    update_message: Optional[str] = None
    accepted: Optional[bool] = None

class ResultArtifact(BaseModel):
    """What result artifact returns after presenting the result."""
    message: Optional[str] = None

llm = LLM(
    model="gpt-4o",
    base_url=OPENAI_API_ENDPOINT,
    api_key=OPENAI_API_KEY,
)

user_representative_agent = Agent(
    role="User Representitive Agent",
    llm=llm,
    goal=(
        "Gather all required information from the user and pass it to the next agent."
        "Present the result of other agents to the user and ask for their feedback"
    ),
    backstory=(
        "You are a user representitive agent part of a crew of agents."
        "The crew is responsible for generating the best joke about the topic."  
        "You need to get topic and number of jokes to select the best from."
        "You will pass this information to the next agent."
        "When other agents returns a result present it to the user and ask for their feedback"
        "User can either accept the result or want to change something."
        "You will NEVER generate jokes yourself, only gather information and present result."   
    ),
    allow_delegation=True,
    tools=[],
    trace=True,
    # goal=(
    #     "You are the ONLY user-facing agent. "
    #     "First greet the user and then ask for the topic and number of jokes to select the best from. "
    #     "Mode 1 (Collect): Ask questions until both the topic and number of jokes are provided."
    #     "Return ONLY JSON as UserInputs: {topic: .., num_jokes: ..}. "
    #     "Mode 2 (Present): Given a ResultArtifact, create a clear, concise user-facing message and ask the user if they accept it or wants to change something. "
    #     "Return ONLY JSON as Presentation: {message: str}. "
    #     "Mode 3 (Iterate or exit): Determine if the user satisfied with the result or wants to change something. "
    #     "Return ONLY JSON as UserResponse: {update_message: str, accepted: bool}. "
    #     "NEVER decide next steps; NEVER run tools."
    # ),
    
)

joke_generator_agent = Agent(
    role="Joke Generator Agent",
    llm=llm,
    goal=(
        "Generate a list of jokes about the topic."
    ),
    backstory=(
        "You are a joke generator agent part of a crew of agents."
        "The overall goal is to pick the best joke from the list of jokes."
        "You need to generate a list of jokes about the topic."
        "Be funny and creative"
        "You will return the list of jokes to the next agent that will pick the best joke."
    ),
    allow_delegation=True,
)

joke_selector_agent = Agent(
    role="Joke Selector Agent",
    llm=llm,
    goal=(
        "Pick the best joke from the list of jokes."
    ),
    backstory=(
        "You are a joke selector agent part of a crew of agents."
        "The overall goal is to pick the best joke from the list of jokes."
        "You need to pick the best joke from the list of jokes."
        "You need also provide reasoning for your decision." 
        "You will return the best joke and your reasoning to the next agent that will present it to the user."
    ),
    allow_delegation=True,
)

def start_chat_loop():
    """Start the chat loop."""

    crew = Crew(
        agents=[user_representative_agent, joke_generator_agent, joke_selector_agent],
        tasks=[
            Task(
                name="start_chat", 
                description="Chat with the user and gather information about the topic and number of jokes to generate the best joke.",
                expected_output="A JSON object with topic and num_jokes fields containing the user's input",
                agent=user_representative_agent
            ),
            Task(
                name="generate_jokes", 
                description="Generate a list of jokes about the topic.",
                expected_output="A list of creative and funny jokes about the specified topic",
                agent=joke_generator_agent
            ),
            Task(
                name="select_best_joke", 
                description="Pick the best joke from the list of jokes.",
                expected_output="The best joke selected from the list along with reasoning for the selection",
                agent=joke_selector_agent
            ),
            Task(
                name="present_result", 
                description="Present the result of the best joke to the user.",
                expected_output="A user-friendly presentation of the best joke with the reasoning",
                agent=user_representative_agent
            ),
        ],
        verbose=True,
    )
    return crew.kickoff()

def test_chat_loop():
    """Test the chat loop."""

    history = []

    chat_agent = Agent(
        role="Chat Agent",
        llm=llm,
        goal="Chat with the user and gather information about the topic and number of jokes to generate the best joke.",
        backstory="You talk to the user and gather information about the topic and number of jokes to generate the best joke.",
        allow_delegation=True,
    )

    joke_task = Task(
        name="generate_jokes", 
        agent=joke_generator_agent,
        expected_output="A list of jokes about the topic",
        description="Generate a list of jokes about the topic."
    )

    while True:
        history_text = "\n".join(history)

        chat_task = Task(
            name="start_chat", 
            agent=chat_agent,
            expected_output="A string: welcome message only for the first message; the the next question to be asked; summary and message that work in progress", 
            description=f"Check history to see if you have all the information you need. If you have all information delegate to the next agent. \nHistory: {history_text}")

        crew = Crew(
            agents=[chat_agent, joke_generator_agent],
            tasks=[chat_task, joke_task],
            verbose=True,
        )
        result = crew.kickoff()
        if result.raw is not None:
            history.append("AI: " + result.raw)
        print(history[-1])

        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit", "bye"}:
            break
        if not user_input:
            continue
        history.append("User: " + user_input)


async def main():
    """Main function - entry point of the program"""
    print("Starting LangGraph joke workflow...")
    test_chat_loop()    
    print("Main program completed successfully!")
    
if __name__ == "__main__":
    asyncio.run(main())