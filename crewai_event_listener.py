from crewai.events import (
    CrewKickoffStartedEvent,
    CrewKickoffCompletedEvent,
    AgentExecutionCompletedEvent,
    FlowFinishedEvent,
    FlowStartedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)
from crewai.events import BaseEventListener

class CrewAiCustomListener(BaseEventListener):
    def __init__(self):
        super().__init__()

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source, event):
            print(f"Crew '{event.crew_name}' has started execution!")

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source, event):
            print(f"Crew '{event.crew_name}' has completed execution!")
            print(f"Output: {event.result}")

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_execution_completed(source, event):
            print(f"Agent '{event.agent.role}' completed task")
            print(f"Output: {event.result}")

        @crewai_event_bus.on(FlowStartedEvent)
        def on_flow_started(source, event):
            print(f"Flow '{event.flow_name}' has started execution!")

        @crewai_event_bus.on(FlowFinishedEvent)
        def on_flow_finished(source, event):
            print(f"Flow '{event.flow_name}' has completed execution!")
            print(f"Output: {event.result}")

        @crewai_event_bus.on(MethodExecutionStartedEvent)
        def on_method_execution_started(source, event):
            print(f"Method '{event.method_name}' has started execution!")

        @crewai_event_bus.on(MethodExecutionFinishedEvent)
        def on_method_execution_finished(source, event):
            print(f"Method '{event.method_name}' has completed execution!")
            print(f"Output: {event.result}")