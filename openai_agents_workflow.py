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
    instructions="You are a orchestrator agent. Your overall goal is to generate the best joke about the topic. You are responsible for coordinating the workers and the decision maker. User will provide the topic of the joke and number of jokes to pick the best from. Use tools to coordinate the workers and the decision maker.",
    model=model,
    tools=[joke_generator_tool, decision_maker_agent.as_tool(
        tool_name="decision_maker",
        tool_description="Use this tool to pick the best joke from the list of jokes. As well as reason for your decision.",
    )]
)

user_chat_agent = Agent(
    name="user_chat",
    instructions="You are a user chat agent helping user to generate the best joke about the topic. You need to get information about jokes to be generated. You will neer to get the topic and number of jokes to be generated. When you have these information hand over to the orchestrator agent. First welcome the user and then ask for the topic and number of jokes to be generated.",
    model=model,
    handoffs=[orchestrator_agent]
)
    

async def generate_best_joke(input: str) -> str:
    """
    Generate the best joke about the topic.
    """
    best_joke = await Runner.run(orchestrator_agent, input=input)
    return best_joke.final_output

async def start_chat_loop() -> str:
    input_items: list[TResponseInputItem] = []
    welcome = False
    while True:
        if welcome:
            user_input = input(" > ")
            if user_input.strip().lower() in {"exit", "quit", "bye"}:
                break
            if not user_input:
                continue
            input_items.append({"role": "user", "content": user_input})
        result = await Runner.run(user_chat_agent, input_items, run_config=RunConfig(tracing_disabled=True))
        if result.final_output is not None:
            if not welcome:
                welcome = True
            input_items.extend(result.to_input_list())
            print(result.final_output)
            # if result.last_agent == orchestrator_agent:
            #     break

def test_function(input: str) -> str:
    """
    Test function that prints a simple message
    """

    agent = Agent(
        name="test_agent",
        instructions="You are helpful assistant." ,
        tools=[],
        model=model,
   )
    result  = Runner.run_sync(agent, input=input)
    return result.final_output


async def main():
    """
    Main function - entry point of the program
    """
    print("Starting main program...")
    
    # Call the test function
    # result = await generate_best_joke("Give the best from five about the dogs.")
    # print(result)
    result = await start_chat_loop()
    print(result)

    # await run_demo_loop(user_chat_agent)

    print("Main program completed successfully!")


if __name__ == "__main__":
    # This allows the script to be run directly
    # and also makes it debuggable
    asyncio.run(main())
