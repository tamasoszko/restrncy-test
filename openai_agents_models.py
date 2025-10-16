### Data Models

from agents import TResponseInputItem
from pydantic import BaseModel, Field
from litellm import dataclass

class ChatSummaryData(BaseModel):
    type_of_cuisine: str = Field(description="The type of cuisine the user is looking for")
    price_range: str = Field(description="The price range the user is looking for")
    user_request_summary: str = Field(description="Overall summary of the user request including all the relevant information")
    best_restaurant: str | None = Field(description="The latest best restaurant that was presented to the user previously, optional.", default=None)
    user_feedback_summary: str | None = Field(description="Summarize the user feddback about the most recent result, ONLY for the most recent result, optional.", default=None)
    handoff_reason: str  = Field(description="Explain why you are handing off to the orchestrator")

class ResultData(BaseModel):
    best_restaurant: str
    decision_reason: str
    all_restaurants: list[str]


class WorflowContext(BaseModel):  
    session_id: str
    chat_history: list[TResponseInputItem]
    chat_summary: ChatSummaryData | None
    last_result: ResultData | None