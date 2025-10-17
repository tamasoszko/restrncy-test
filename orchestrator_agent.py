from agents import Agent
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from decision_maker_agent import decision_maker_agent
from inquiry_agent import restaurant_inquiry_tool
from models import default_model
from openai_agents_utils import LoggerHooks



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
    model=default_model,
    hooks=LoggerHooks(),
    tools=[restaurant_inquiry_tool, 
        decision_maker_agent.as_tool(
            tool_name="decision_maker",
            tool_description="Use this tool to pick the best restaurant from the list of restaurants. As well as reason for your decision.",
        ),
    ],
)