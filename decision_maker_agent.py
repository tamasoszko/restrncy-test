from agents import Agent
from models import default_model


decision_maker_agent = Agent(
    name="decision_maker",
    instructions="You are a decision maker agent. Pick the best restaurant from the list of restaurants based on the user request.",
    model=default_model 
)