#!/usr/bin/env python3
"""
Main Python file - executable and debuggable
"""

import asyncio
from typing import Any
import dotenv
import os
from agents import Agent, ModelSettings, RunConfig, Runner, TResponseInputItem, Trace, function_tool, set_trace_processors, set_tracing_disabled, trace, run_demo_loop
from agents.extensions.models.litellm_model import LitellmModel
from agents.extensions.visualization import draw_graph
import mlflow


dotenv.load_dotenv()


OPENAI_API_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
MLFLOW_TRACING_URL = os.getenv("MLFLOW_TRACING_URL")

if MLFLOW_TRACING_URL is not None:
    mlflow.openai.autolog()
    mlflow.set_tracking_uri(MLFLOW_TRACING_URL)
    mlflow.set_experiment("OpenAI Agent")


model = LitellmModel( model=OPENAI_MODEL_NAME, api_key=OPENAI_API_KEY, base_url=OPENAI_API_ENDPOINT)


# Disable default tracing

set_tracing_disabled(True)


# Tools

@function_tool
async def joke_generator_tool(topic: str, num_jokes: int = 3) -> str:
    """
    Joke generator function. Generate a list of jokes about the topic.

    Args:
        topic: The topic of the joke
        num_jokes: The number of jokes to generate
    """ 
    
    runs = []
    for i in range(num_jokes):
        agent = worker_agent.clone(
            name=f"worker_{i}",
        )
        runs.append(Runner.run(agent, input=topic))
    joke_results = await asyncio.gather(*runs)
    jokes = "\n----------\n".join([joke.final_output for joke in joke_results])
    print("All jokes: \n\n")
    for i, joke in enumerate(joke_results):
        print(f"{i+1}: {joke.final_output}\n")
    print("\n\n")


    return jokes

# Agents

decision_maker_agent = Agent(
    name="decision_maker",
    instructions="You are a decision maker agent. Pick the best joke from the list of jokes. Always provide the reason for your decision as well.",
    model=model 
)

worker_agent = Agent(
    name="worker",
    instructions="Give a short joke about the topic. Be creative and funny.",
    model=model,
    model_settings=ModelSettings(
        temperature=0.5
    ),
)

orchestrator_agent = Agent(
    name="orchestrator",
    instructions=(
        "You are a orchestrator agent. Your overall goal is to generate the best joke about the topic."
        "Begin EVERY message with '[Orchestrator]: ' "
        "You are responsible for coordinating the workers, decision maker and presenter."
        "You are not generating jokes yourself, you will only coordinate other agents through tools."
        "You will be provided with the topic of the joke and number of jokes to pick the best from then use the tools to generate the best joke."
        "When you have the best joke, hand off to the user_chat_agent" 
        "User may want to change something or want to generate new jokes. Perform the same process again." 
        ),
    model=model,
    tools=[joke_generator_tool, 
        decision_maker_agent.as_tool(
            tool_name="decision_maker",
            tool_description="Use this tool to pick the best joke from the list of jokes. As well as reason for your decision.",
        ),
    ],
)

user_chat_agent = Agent(
    name="user_chat",
    instructions=(
        "You are a user chat agent helping user to generate the best joke about the topic."
        "Begin EVERY message with '[Assistant]: ' "
        "You have two responsibilities:"
        "1. You need to get information about jokes to be generated. You will neer to get the topic and number of jokes to be generated. "
        "2. You present the final result -the best joke and the reasoning- to the user and ask if they are satisfied with the result or wants some changes."
        "If the use is satisfied, say goodbye and tell them to type 'exit' or 'bye' to end the conversation."
        "You never generate jokes yourself, you will only talk to the user understand what they and if there is enough information hand over to the orchestrator agent." 
        "At first welcome the user and then ask for the topic and number of jokes to be generated." 
    ),
    model=model,
    handoffs=[orchestrator_agent]
)
    

async def start_chat_loop() -> str:
    input_items: list[TResponseInputItem] = []
    while True:
        user_input = input("[User]: ")
        if user_input.strip().lower() in {"exit", "quit", "bye", "bb", "q"}:
            break
        if not user_input:
            continue
        input_items.append({"role": "user", "content": user_input})
        orchestrator_agent.handoffs.append(user_chat_agent)
        result = await Runner.run(user_chat_agent, input_items, run_config=RunConfig(tracing_disabled=False))
        if result.final_output is not None:
            input_items = result.to_input_list()
            print(result.final_output)


async def main():
    """
    Main function - entry point of the program
    """
    print("Starting main program...")  
    draw_graph(user_chat_agent, "openai_agents_workflow")

    with mlflow.start_run(run_name="joke_generator"):
        result = await start_chat_loop()
        print(result)

    print("Main program completed successfully!")


if __name__ == "__main__":
    # This allows the script to be run directly
    # and also makes it debuggable
    asyncio.run(main())
