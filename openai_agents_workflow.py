#!/usr/bin/env python3
"""
Main Python file - executable and debuggable
"""

import asyncio
import dotenv
import os
from agents import Agent, HandoffInputData, ModelResponse, ModelSettings, RunConfig, RunContextWrapper, RunItem, Runner, TResponseInputItem, function_tool, handoff, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from agents.extensions.visualization import draw_graph
from litellm import dataclass
import mlflow
from pydantic import BaseModel
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from agents.extensions import handoff_filters

from openai_agents_utils import LoggerHooks, history_item_to_string

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

class ResultData(BaseModel):
    best_joke: str
    decision_reason: str
    all_jokes: list[str]

@dataclass
class WorflowContext:  
    name: str
    chat_history: list[TResponseInputItem]
    last_result: ResultData | None

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
    print("[Trace] All jokes: \n\n")
    for i, joke in enumerate(joke_results):
        print(f"[Trace] {i+1}: {joke.final_output}\n")
    print("[Trace] \n\n")

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
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are a orchestrator agent. Your overall goal is to generate the best joke about the topic."
        "Begin EVERY message with '[Orchestrator]: ' "
        "You are responsible for coordinating the workers, decision maker and presenter."
        "You are not generating jokes yourself, you will only coordinate other agents through tools."
        "You will be provided with the topic of the joke and number of jokes to pick the best from then use the tools to generate the best joke."
        "When you have the best joke, hand off to the user_chat_agent" 
        "User may want to change something or want to generate new jokes. Perform the same process again." 
        ),
    model=model,
    hooks=LoggerHooks(),
    tools=[joke_generator_tool, 
        decision_maker_agent.as_tool(
            tool_name="decision_maker",
            tool_description="Use this tool to pick the best joke from the list of jokes. As well as reason for your decision.",
        ),
    ],
)

chat_summary_agent = Agent(
    name="chat_summary",
    instructions=(
        "You are a chat summarizer agent."
        "Begin EVERY message with '[Chat Summarizer]: ' "
        "You will be provided with the chat history and you will need to summarize it."
        "Generate the summary in the following format:\n"
        " - topic: <topic> the user requested, should be always present in history, use the last one"
        " - number of jokes: <number of jokes> the user requested, should be always present in history, use the last one"
        " - best joke: <best joke> optional, if it is present in the chat history"
        " - user feedback: <user feedback about the best joke if it was presented> optional, if it is present in the chat history, use the last one"
    ),
    model=model,
)

# Handoff handlers: User Chat -> Orchestrator
# remove chat history and add summary as history
# - call chat_summary_agent to summarize the chat history
# - save chat history to context
# - return only the summary as history

async def on_handoff_user_chat_to_orchestrator(ctx: RunContextWrapper[WorflowContext]):
    print(f"[Trace] ctx={ctx}")

async def handoff_filter_user_chat_to_orchestrator(input: HandoffInputData) -> HandoffInputData:
    input = handoff_filters.remove_all_tools(input)    

    history = "\n".join(history_item_to_string(item) for item in input.input_history)

    summary = await Runner.run(chat_summary_agent, history)
    print(f"[Trace] Summary:\n{summary.final_output}")

    run_context = input.run_context
    run_context.context.chat_history = input.input_history

    input_history = summary.to_input_list()[-1:]
    return HandoffInputData(
        input_history=tuple(input_history),
        pre_handoff_items=(),
        new_items=(),
        run_context=run_context,
    )

async def on_handoff_orchestrator_to_user_chat(ctx: RunContextWrapper[WorflowContext], result: ResultData):
    print(f"[Trace] ctx={ctx}")
    ctx.context.last_result = result
    print(f"[Trace] result={result}")


# Handoff handlers: Orchestrator -> User Chat
# rebuild the chat history and add the last result from contextas a developer message:
# - get the last result from context and format it as a developer message
# - get the chat history from context and add the developer message to it
# - return the rebuild chat history as history

async def handoff_filter_orchestrator_to_user_chat(input: HandoffInputData) -> HandoffInputData:
    input = handoff_filters.remove_all_tools(input)    
    
    run_context = input.run_context
    last_message = {"role": "developer", "content": f"{{'best_joke': '{run_context.context.last_result.best_joke}', 'decision_reason': '{run_context.context.last_result.decision_reason}', 'all_jokes': '{run_context.context.last_result.all_jokes}'}}" }
    chat_history = [*input.run_context.context.chat_history, last_message]

    return HandoffInputData(
        input_history=tuple(chat_history),
        pre_handoff_items=(),
        new_items=(),
        run_context=run_context,
    )

user_chat_agent = Agent(
    name="user_chat",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are a user chat agent helping user to generate the best joke about the topic."
        "Begin EVERY message with '[Assistant]: ' "
        "You have two responsibilities:"
        "1. You need to get information about jokes to be generated. You will need to get the topic and number of jokes to be generated. "
        "2. You format and present the final result - { 'best_joke': <best_joke>, 'decision_reason': <decision_reason> }- to the user and ask if they are satisfied with the result or wants some changes."
        "If the use is satisfied, say goodbye and tell them to type 'exit' or 'bye' to end the conversation."
        "You never generate jokes yourself, you will only talk to the user understand what they " 
        "At first welcome the user and then ask for the topic and number of jokes to be generated." 
    ),
    model=model,
    hooks=LoggerHooks(),
    tools=[],
    handoffs=[handoff(
        agent=orchestrator_agent,
        input_filter=handoff_filter_user_chat_to_orchestrator,
        on_handoff=on_handoff_user_chat_to_orchestrator,
    )]
)
    


async def start_chat_loop() -> str:
    input_items: list[TResponseInputItem] = []
    ctx = WorflowContext(name="user_chat", chat_history=[], last_result=None)
    while True:
        user_input = input("[User]: ")
        if user_input.strip().lower() in {"exit", "quit", "bye", "bb", "q"}:
            break
        if not user_input:
            continue
        input_items.append({"role": "user", "content": user_input})
        # add handoff to user chat agent here to avoid circular dependency
        orchestrator_agent.handoffs.append(handoff(
            agent=user_chat_agent,
            input_type=ResultData,
            input_filter=handoff_filter_orchestrator_to_user_chat,
            on_handoff=on_handoff_orchestrator_to_user_chat,
        ))
        result = await Runner.run(user_chat_agent, input_items, run_config=RunConfig(tracing_disabled=False), context=ctx)
        if result.final_output is not None:
            input_items = result.to_input_list()
            print(result.final_output)


async def main():
    """
    Main function - entry point of the program
    """
    print("[Trace] Starting main program...")  
    draw_graph(user_chat_agent, "openai_agents_workflow")

    with mlflow.start_run(run_name="joke_generator"):
        result = await start_chat_loop()
        print(result)

    print("[Trace] Main program completed successfully!")


if __name__ == "__main__":
    # This allows the script to be run directly
    # and also makes it debuggable
    asyncio.run(main())
