#!/usr/bin/env python3
"""
Main Python file - executable and debuggable
"""

import asyncio
import uuid
from agents import HandoffInputData, RunConfig, RunContextWrapper, Runner, handoff
import mlflow
from agents.extensions import handoff_filters

from openai_agents_models import ChatSummaryData, ResultData, WorflowContext
from openai_agents_session_manager import SessionManager
from orchestrator_agent import orchestrator_agent
from user_chat_agent import user_chat_agent
from tracing import init_mlflow_tracing


init_mlflow_tracing()


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

class RecommenderWorkflow:
    def __init__(self, session_id: str | None):
        self.session_id = session_id if session_id is not None else str(uuid.uuid4())
        self.user_chat_agent = user_chat_agent


    @staticmethod
    def setup():
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

    async def resume(self, user_input: str) -> tuple[str, str]:
        ctx = WorflowContext(session_id=self.session_id, chat_history=[], last_result=None, chat_summary=None)
        session_manager = SessionManager(session_id=self.session_id)
        session = session_manager.get_session(agent=user_chat_agent)
        result = await Runner.run(user_chat_agent, user_input, run_config=RunConfig(tracing_disabled=False), context=ctx, session=session)
        return session_manager.session_id, result.final_output


async def start_chat_loop(session_id: str | None = None) -> str:

    workflow = RecommenderWorkflow(session_id=session_id)

    while True:
        user_input = input("[User]: ")
        if user_input.strip().lower() in {"exit", "quit", "bye", "bb", "q"}:
            break
        result = await workflow.resume(user_input)
        print(result)

async def main():
    """
    Main function - entry point of the program
    """
    print("[Trace] Starting main program...")  
    RecommenderWorkflow.setup()

    with mlflow.start_run(run_name=f"restaurant_finder"):
        result = await start_chat_loop()
        print(result)

    print("[Trace] Main program completed successfully!")


if __name__ == "__main__":
    # This allows the script to be run directly
    # and also makes it debuggable
    asyncio.run(main())
