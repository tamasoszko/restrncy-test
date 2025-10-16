### Data Models

from agents import TResponseInputItem
from pydantic import BaseModel, Field
from litellm import dataclass

class ChatSummaryData(BaseModel):
    topic: str = Field(description="The topic of the joke")
    number_of_jokes: int = Field(description="The number of jokes to generate")
    best_joke: str | None = Field(description="The latest best joke that was presented to the user previously, optional.", default=None)
    user_feedback_summary: str | None = Field(description="Summarize the user feddback about the most recent result, ONLY for the most recent result, optional.", default=None)
    handoff_reason: str  = Field(description="Explain why you are handing off to the orchestrator")

class ResultData(BaseModel):
    best_joke: str
    decision_reason: str
    all_jokes: list[str]


class WorflowContext(BaseModel):  
    session_id: str
    chat_history: list[TResponseInputItem]
    chat_summary: ChatSummaryData | None
    last_result: ResultData | None