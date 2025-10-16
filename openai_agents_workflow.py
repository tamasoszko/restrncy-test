#!/usr/bin/env python3
"""
Main Python file - executable and debuggable
"""

import asyncio
import random
import uuid
import dotenv
import os
from agents import Agent, HandoffInputData, ModelSettings, RunConfig, RunContextWrapper, Runner, function_tool, handoff, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from agents.extensions.visualization import draw_graph
import mlflow
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from agents.extensions import handoff_filters

from openai_agents_models import ChatSummaryData, ResultData, WorflowContext
from openai_agents_utils import LoggerHooks
from openai_agents_session_manager import SessionManager

dotenv.load_dotenv()


OPENAI_API_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
MLFLOW_TRACING_URL = os.getenv("MLFLOW_TRACING_URL")


if MLFLOW_TRACING_URL is not None:
    mlflow.openai.autolog()
    mlflow.set_tracking_uri(MLFLOW_TRACING_URL)
    mlflow.set_experiment("OpenAI Agent")
    print(mlflow.__version__)


model = LitellmModel( model=OPENAI_MODEL_NAME, api_key=OPENAI_API_KEY, base_url=OPENAI_API_ENDPOINT)


# Disable default tracing

set_tracing_disabled(True)


### Tools

@function_tool
async def restaurant_inquiry_tool(context: RunContextWrapper[WorflowContext], user_request: str) -> str:
    """
    Runs inquiry with multiple restaurants in the area.

    Args:
        user_request: The user request for the restaurant inquiry
    """ 
    
    number_of_restaurants = random.randint(2, 5)
    runs = []
    for i in range(number_of_restaurants):
        agent_name = f"worker_{i}"
        session = SessionManager(session_id=context.context.session_id).get_session(agent=agent_name)
        agent = worker_agent.clone(
            name=agent_name,
        )
        runs.append(Runner.run(agent, input=user_request, session=session))

    restaurant_results = await asyncio.gather(*runs)
    restaurants = "\n----------\n".join([restaurant.final_output for restaurant in restaurant_results])
    print("[Trace] All restaurants: \n\n")
    for i, restaurant in enumerate(restaurant_results):
        print(f"[Trace] {i+1}: {restaurant.final_output}\n")
    print("[Trace] \n\n")

    return restaurants


### Agents

decision_maker_agent = Agent(
    name="decision_maker",
    instructions="You are a decision maker agent. Pick the best restaurant from the list of restaurants based on the user request.",
    model=model 
)

worker_agent = Agent(
    name="worker",
    instructions="You are a restaurant representative agent. You are simulating a real restaurant. Answer the user request about the restaurant. Be creative Find out a restaurant name, rating, cuisine, menu, price range, and other relevant information.",
    model=model,
    model_settings=ModelSettings(
        temperature=0.5
    ),
)

orchestrator_agent = Agent(
    name="orchestrator",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are a orchestrator agent. Your overall goal is to pick the best restaurant based on the user request."
        "Begin EVERY message with '[Orchestrator]: ' "
        "You are responsible for coordinating the workers (through restaurant_inquiry_tool), decision maker and chat agent."
        "You are not generating restaurants information yourself, you will only coordinate other agents through tools."
        "You will be provided with the user request then use the tools to find the best restaurant."
        "When you have the best restaurant, hand off to the user_chat_agent" 
        "If the user may want to change something or run a new search for restaurant information -> perform the same process again." 
        ),
    model=model,
    hooks=LoggerHooks(),
    tools=[restaurant_inquiry_tool, 
        decision_maker_agent.as_tool(
            tool_name="decision_maker",
            tool_description="Use this tool to pick the best restaurant from the list of restaurants. As well as reason for your decision.",
        ),
    ],
)

user_chat_agent = Agent(
    name="user_chat",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are a user chat agent helping user to find the best restaurant."
        "Begin EVERY message with '[Assistant]: ' "
        "You have two responsibilities:"
        "1. You need to get information about the user current preferences: type of cuisine and price range"
        "2. You must format and present the result in a user friendly way using markdown (only these fields will be presented): '*Best Restaurant*: {best_restaurant}, *Decision Reason*: {decision_reason}', add emojis to the restaurant to make it more engaging."
        "    Ask if the user is satisfied with the result or wants some changes."
        "    IMPORTANT: Never hand off automatically after this, always check if there was a user input for the last result before handing off. Never hand off if the last message is not from the user."
        "If the use is satisfied, say goodbye and tell them to type 'exit' or 'bye' to end the conversation."
        "You never generate restaurants information yourself, you will only talk to the user understand what they want and present the result given to you." 
        "At first welcome the user and then ask for the type of cuisine and price range." 
    ),
    model=model,
    hooks=LoggerHooks(),
    tools=[],
)

# Handoffs

# User Chat -> Orchestrator: 
# remove chat history and add summary as history:
# - retrieve chat_summary from the context and format it as a developer message
# - save chat history to context
# - return only the summary as history

async def on_handoff_user_chat_to_orchestrator(ctx: RunContextWrapper[WorflowContext], summary: ChatSummaryData):
    print(f"[Trace] summary={summary}")
    ctx.context.chat_summary = summary

async def handoff_filter_user_chat_to_orchestrator(input: HandoffInputData) -> HandoffInputData:
    input = handoff_filters.remove_all_tools(input)    

    run_context = input.run_context
    run_context.context.chat_history = input.input_history

    last_message = {
        "role": "developer", 
        "content": f"""{{
            'chat_summary': '{run_context.context.chat_summary}'
        }}""" 
    }
    input_history = [last_message]
    return HandoffInputData(
        input_history=tuple(input_history),
        pre_handoff_items=(),
        new_items=(),
        run_context=run_context,
    )

# Orchestrator -> User Chat
# rebuild the chat history and add the last result from contextas a developer message:
# - get the last result from context and format it as a developer message
# - get the chat history from context and add the developer message to it
# - return the rebuild chat history as history

async def on_handoff_orchestrator_to_user_chat(ctx: RunContextWrapper[WorflowContext], result: ResultData):
    print(f"[Trace] result={result}")
    ctx.context.last_result = result

async def handoff_filter_orchestrator_to_user_chat(input: HandoffInputData) -> HandoffInputData:
    input = handoff_filters.remove_all_tools(input)    
    
    run_context = input.run_context
    last_message = {
        "role": "developer", 
        "content": f"""{{
            'best_restaurant': '{run_context.context.last_result.best_restaurant}', 
            'decision_reason': '{run_context.context.last_result.decision_reason}', 
            'all_restaurants': '{run_context.context.last_result.all_restaurants}'
        }}""" 
    }
    chat_history = [*input.run_context.context.chat_history, last_message]

    return HandoffInputData(
        input_history=tuple(chat_history),
        pre_handoff_items=(),
        new_items=(),
        run_context=run_context,
    )


async def start_chat_loop() -> str:

    # wire handoff programatically to avoid circular dependency
    user_chat_agent.handoffs.append(handoff(
        agent=orchestrator_agent,
        input_type=ChatSummaryData,
        input_filter=handoff_filter_user_chat_to_orchestrator,
        on_handoff=on_handoff_user_chat_to_orchestrator,
    ))

    orchestrator_agent.handoffs.append(handoff(
        agent=user_chat_agent,
        input_type=ResultData,
        input_filter=handoff_filter_orchestrator_to_user_chat,
        on_handoff=on_handoff_orchestrator_to_user_chat,
    ))

    # draw_graph(user_chat_agent, "openai_agents_workflow")

    session_id = str(uuid.uuid4())

    while True:
        ctx = WorflowContext(session_id=session_id, chat_history=[], last_result=None, chat_summary=None)

        session = SessionManager(session_id=session_id).get_session(agent=user_chat_agent)

        user_input = input("[User]: ")
        if user_input.strip().lower() in {"exit", "quit", "bye", "bb", "q"}:
            break
        if not user_input:
            continue
        result = await Runner.run(user_chat_agent, user_input, run_config=RunConfig(tracing_disabled=False), context=ctx, session=session)
        if result.final_output is not None:
            print(result.final_output)


async def main():
    """
    Main function - entry point of the program
    """
    print("[Trace] Starting main program...")  

    with mlflow.start_run(run_name=f"restaurant_finder"):
        result = await start_chat_loop()
        print(result)

    print("[Trace] Main program completed successfully!")


if __name__ == "__main__":
    # This allows the script to be run directly
    # and also makes it debuggable
    asyncio.run(main())
