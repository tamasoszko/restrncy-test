
from typing import Any
from agents import Agent, AgentHooks, ModelResponse, RunContextWrapper, TResponseInputItem
import mlflow
from mlflow.entities import SpanType
import mlflow
from typing import Any


def history_item_to_string(item: str | TResponseInputItem) -> str:

    def _content_item_to_string(content: any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, dict) and "text" in content:
            return content['text']
        return str(content)

    if isinstance(item, str):
        return item
    if "role" not in item or "content" not in item:
        return str(item)
    content = item["content"]
    if isinstance(content, str):
        return f"{item['role']}: {content}"
    if isinstance(content, list):
        content_str = "\n".join(_content_item_to_string(item) for item in content)
        return f"{item['role']}: {content_str}"
    return str(content)


class LoggerHooks(AgentHooks):

    def __init__(self, log_on_handoff: bool = True, 
                        log_on_llm_start: bool = False, 
                        log_on_llm_end: bool = False, 
                        log_on_agent_start: bool = False, 
                        log_on_agent_end: bool = False):
        super().__init__()
        self.log_on_handoff = log_on_handoff
        self.log_on_llm_start = log_on_llm_start
        self.log_on_llm_end = log_on_llm_end
        self.log_on_agent_start = log_on_agent_start
        self.log_on_agent_end = log_on_agent_end

    async def on_handoff(self, context: RunContextWrapper[None], agent: Agent[None], source: Agent[None]):
        if self.log_on_handoff:
            with mlflow.start_span(name=f"Handoff_{source.name}_to_{agent.name}", span_type=SpanType.CHAIN) as span:
                span.set_inputs({"context": str(context.context)})

            print(f"[Event] on_handoff: '{source.name}' -> '{agent.name}'")

    async def on_llm_start(self, context: RunContextWrapper[None], agent: Agent[None], system_prompt: str, input_items: list[TResponseInputItem]):
        if self.log_on_llm_start:
            print(f"[Event] on_llm_start: '{agent.name}', input_items: '{input_items}'")

    async def on_llm_end(self, context: RunContextWrapper[None], agent: Agent[None], response: ModelResponse):
        if self.log_on_llm_end:
            print(f"[Event] on_llm_end: '{agent.name}', output: '{response.final_output}'")

    async def on_agent_start(self, context: RunContextWrapper[None], agent: Agent[None]):
        if self.log_on_agent_start:
            print(f"[Event] on_agent_start: '{agent.name}'")

    async def on_agent_end(self, context: RunContextWrapper[None], agent: Agent[None], output: Any):
        if self.log_on_agent_end:
            print(f"[Event] on_agent_end: '{agent.name}'")