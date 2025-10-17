from agents import Agent
from models import default_model
from agents.extensions.models.litellm_model import ModelSettings
from agents import function_tool, RunContextWrapper
from openai_agents_models import WorflowContext
from openai_agents_session_manager import SessionManager
import random
import asyncio
from agents import Runner

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
        agent = inquiry_agent.clone(
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


inquiry_agent = Agent(
    name="worker",
    instructions="You are a restaurant representative agent. You are simulating a real restaurant. Answer the user request about the restaurant. Be creative Find out a restaurant name, rating, cuisine, menu, price range, and other relevant information.",
    model=default_model,
    model_settings=ModelSettings(
        temperature=0.5
    ),
)