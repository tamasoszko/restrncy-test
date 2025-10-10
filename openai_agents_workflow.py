#!/usr/bin/env python3
"""
Main Python file - executable and debuggable
"""

import asyncio
import dotenv
import os
from agents import Agent, ModelSettings, RunConfig, Runner, TResponseInputItem, function_tool, trace, run_demo_loop
from agents.extensions.models.litellm_model import LitellmModel

dotenv.load_dotenv()

OPENAI_API_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = LitellmModel(model="gpt-4o", api_key=OPENAI_API_KEY, base_url=OPENAI_API_ENDPOINT)

# Tools

@function_tool
async def joke_generator_tool(topic: str, num_jokes: int = 3) -> str:
    """
    Joke generator function. Generate a list of jokes about the topic.

    Args:
        topic: The topic of the joke
        num_jokes: The number of jokes to generate
    """ 
    with trace("joke_generator"):
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
    welcome = False
    while True:
        if welcome:
            user_input = input("[User]: ")
            if user_input.strip().lower() in {"exit", "quit", "bye"}:
                break
            if not user_input:
                continue
            input_items.append({"role": "user", "content": user_input})
        orchestrator_agent.handoffs.append(user_chat_agent)
        result = await Runner.run(user_chat_agent, input_items, run_config=RunConfig(tracing_disabled=True))
        if result.final_output is not None:
            if not welcome:
                welcome = True
            input_items.extend(result.to_input_list())
            print(result.final_output)
            # if result.last_agent == orchestrator_agent:
            #     break

async def main():
    """
    Main function - entry point of the program
    """
    print("Starting main program...")    
    result = await start_chat_loop()
    print(result)


    print("Main program completed successfully!")


if __name__ == "__main__":
    # This allows the script to be run directly
    # and also makes it debuggable
    asyncio.run(main())
