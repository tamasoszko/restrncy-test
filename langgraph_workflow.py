#!/usr/bin/env python3
"""
LangGraph implementation of the OpenAI agents workflow
"""

import asyncio
import dotenv
import os
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langchain_core.outputs import chat_result
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
import operator

from pydantic import BaseModel, Field

# Load environment variables
dotenv.load_dotenv()

OPENAI_API_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class UserChat(BaseModel):
    """What chat agent returns after asking for missing information."""
    agent_response: Optional[str] = Field(default=None)
    topic: Optional[str] = Field(default=None)
    num_jokes: Optional[int] = Field(default=None)
    chat_result: Optional[str] = Field(default=None) # "user_input", "to_generate", "finished", "update"

class Decision(BaseModel):
    """What decision maker returns after picking the best joke."""
    best_joke: Optional[str] = Field(default=None)
    decision_reason: Optional[str] = Field(default=None)

# Initialize the model
joke_llm = ChatOpenAI(
    model="gpt-4o",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_ENDPOINT,
    temperature=0.7
)

generic_llm = ChatOpenAI(
    model="gpt-4o",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_ENDPOINT,
    temperature=0.0
)

class JokeWorkflowState(TypedDict):
    """State for the joke generation workflow"""
    messages: Annotated[List, operator.add]  # Chat messages
    topic: Optional[str]  # Joke topic
    num_jokes: Optional[int]  # Number of jokes to generate
    jokes: List[str]  # Generated jokes
    best_joke: Optional[str]  # Selected best joke
    decision_reason: Optional[str]  # Reason for selection
    conversation_stage: str  # "welcome", "user_input", "generating", "deciding", "presenting", "complete"
    user_input: Optional[str]  # Current user input

# Tools
@tool
def generate_joke_worker(topic: str) -> str:
    """Generate a single joke about the given topic. Be creative and funny."""

    system_message = SystemMessage(content= f"""You are a joke generator. Generate a single joke about the given topic. Be creative and funny.""")
    result = joke_llm.invoke([system_message, HumanMessage(content=topic)])
    return result.content


# Node functions
def user_chat_node(state: JokeWorkflowState) -> JokeWorkflowState:
    """Handle user interaction and collect requirements"""
    messages = state.get("messages", [])
    conversation_stage = state.get("conversation_stage")

    system_message = { "role": "system", "content": 
        """You are chat agent. Your task is to help user to generate the best joke about the topic. 
        You will NOT generate jokes yourself, you will only talk to the user understand what they want and present the result given to you.
        The best joke will be selected by other agents based on you input and you will the result. 
        Begin with welcoming the user and then ask for the topic and number of jokes to pick the best from.
        Keep chatting with the user until you have both the information you need.
        If you have the best joke, you present is to the user along with the reason for the decision
        and ask the user if they accept it or wants to change something. If the user want something new 
        'chat_result' can be 
        - 'user_input' if you are asking for the topic and number of jokes or when you are presenting the result and asking for the user to accept it, 
        - 'to_generate' if all information is gathered in order to initially generate the best joke 
        - 'finished' ONLY if the user accepted the result of the best joke.
        - 'update' when result is presented and the user wants to modify something or generate new jokes.
        'agent_response' is the response from you that will be shown to the user.
        Return ONLY JSON matching the UserChat schema.""" }


    if conversation_stage == "user_input":
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            return state
        if not user_input:
            return state
        messages.append(HumanMessage(content=user_input))
    elif conversation_stage == "presenting" and state.get("best_joke") is not None:
        result_message = AIMessage(
            content= f"""best joke: {state.get("best_joke")}
            Reason for your decision: {state.get("decision_reason")}
            """
        )
        messages.append(result_message)

    llm = generic_llm.with_structured_output(UserChat)
    response = llm.invoke([
        system_message,
        *messages,
    ])
    if response.agent_response is not None:
        messages.append(AIMessage(content=response.agent_response))
        print(response.agent_response)
    conversation_stage = "user_input"
    if response.chat_result == "finished":
        conversation_stage =  "complete"
    elif response.chat_result == "update":
        conversation_stage = "generating"
    elif response.chat_result == "to_generate":
        conversation_stage = "generating"
    
    state["conversation_stage"] = conversation_stage
    state["messages"] = messages
    state["topic"] = response.topic
    state["num_jokes"] = response.num_jokes
    return state

def joke_generation_node(state: JokeWorkflowState) -> JokeWorkflowState:
    """Generate multiple jokes in parallel"""
    topic = state.get("topic")
    num_jokes = state.get("num_jokes", 3)
    
    if not topic:
        return state
    
    # Generate jokes using the tool
    jokes = []
    for i in range(num_jokes):
        joke = generate_joke_worker.invoke({"topic": topic})
        jokes.append(joke)
    
    messages = state.get("messages", [])
    jokes_text = "\n".join([f"{i+1}. {joke}" for i, joke in enumerate(jokes)])
    generation_msg = f"Here are the {num_jokes} jokes I generated:\n\n{jokes_text}"
    messages.append(AIMessage(content=generation_msg))
    
    return {
        "messages": messages,
        "jokes": jokes,
        "conversation_stage": "deciding"
    }

def decision_making_node(state: JokeWorkflowState) -> JokeWorkflowState:
    """Select the best joke and provide reasoning"""
    jokes = state.get("jokes", [])
    topic = state.get("topic", "")
    
    if not jokes:
        return state
    
    # Create a prompt for the decision maker
    jokes_text = "\n----------\n".join([f"{i+1}. {joke}" for i, joke in enumerate(jokes)])
    system_message = SystemMessage(content= f"""You are a decision maker. Pick the best joke from the following list about "{topic}".
        Please select the best joke and provide your reasoning. 
        Return ONLY JSON matching the Decision schema.
        """)

    human_message = HumanMessage(content=jokes_text)
    # Get decision from the model
    llm = generic_llm.with_structured_output(Decision)
    response = llm.invoke([system_message, human_message])
        
    return {
        "best_joke": response.best_joke,
        "decision_reason": response.decision_reason,
        "conversation_stage": "presenting"
    }

def should_continue(state: JokeWorkflowState) -> str:
    """Determine the next step in the workflow"""
    stage = state.get("conversation_stage", "welcome")
    
    if stage == "generating":
        return "joke_generation"
    elif stage == "deciding":
        return "decision_making"
    elif stage == "complete":
        return END
    else:
        return "user_chat"

def build_joke_workflow_graph():
    """Build the LangGraph workflow"""
    workflow = StateGraph(JokeWorkflowState)
    
    # Add nodes
    workflow.add_node("user_chat", user_chat_node)
    workflow.add_node("joke_generation", joke_generation_node)
    workflow.add_node("decision_making", decision_making_node)
    
    # Add edges
    workflow.add_edge(START, "user_chat")
    workflow.add_conditional_edges(
        "user_chat",
        should_continue,
        {
            "user_chat": "user_chat",
            "joke_generation": "joke_generation",
            "decision_making": "decision_making",
            END: END
        }
    )
    workflow.add_conditional_edges(
        "joke_generation",
        should_continue,
        {
            "decision_making": "decision_making",
            END: END
        }
    )
    workflow.add_conditional_edges(
        "decision_making",
        should_continue,
        {
            "user_chat": "user_chat",
            END: END
        }
    )
    
    return workflow.compile()

async def start_chat_loop():
    """Start the interactive chat loop"""
    graph = build_joke_workflow_graph()
    
    # Initialize state
    state = {
        "messages": [],
        "topic": None,
        "num_jokes": None,
        "jokes": [],
        "best_joke": None,
        "decision_reason": None,
        "conversation_stage": "welcome",
        "user_input": None
    }
    
    print("Starting joke generation workflow...")
    print("Type 'exit', 'quit', or 'bye' to end the conversation.\n")
    
    while True:
        # Run the graph
        result = await graph.ainvoke(state)
        
        # Print the latest AI message
        if result["messages"]:
            latest_message = result["messages"][-1]
            if isinstance(latest_message, AIMessage):
                print(latest_message.content)
                print()
        
        # Check if conversation is complete
        if result.get("conversation_stage") == "complete":
            break
        
        # Get user input
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Update state with user input
        state = result.copy()
        state["user_input"] = user_input


async def main():
    """Main function - entry point of the program"""
    print("Starting LangGraph joke workflow...")
    await start_chat_loop()    
    print("Main program completed successfully!")

if __name__ == "__main__":
    # This allows the script to be run directly
    # and also makes it debuggable
    asyncio.run(main())