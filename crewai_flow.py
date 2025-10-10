import asyncio
import random
from tkinter import NO
from typing import Any
import uuid


from crewai.flow import or_, router
from crewai.flow.flow import Flow, listen, start
from crewai.flow.persistence.decorators import persist
import openlit

from crewai_event_listener import CrewAiCustomListener
from crewai_persistance import InMemoryFlowPersistence

openlit.init(disable_metrics=True, otlp_endpoint="http://127.0.0.1:4318")

event_listener = CrewAiCustomListener()
persistence = InMemoryFlowPersistence()

@persist(persistence)  # Using InMemoryFlowPersistence instance
# @persist()
class OutputExampleFlow(Flow):        

    @start()
    def initialize(self):
        if "counter" not in self.state:
            self.state["counter"] = 0
        if "user_inputs" not in self.state:
            self.state["user_inputs"] = []
        if "retry_count" not in self.state:
            self.state["retry_count"] = 2
        if "attempt_count" not in self.state:
            self.state["attempt_count"] = 1
        if "output_messages" not in self.state:
            self.state["output_messages"] = []
        if "history" not in self.state:
            self.state["history"] = []

    @listen(initialize)
    def handle_user_input(self) -> str | None:
        if "last_user_input" in self.state:
            user_input = self.__pop_from_state(key="last_user_input")
            input_type = self.__pop_from_state(key="user_input_type", default="user_inputs")
            if len(user_input) == 0:
                return None
            if input_type == "user_choice":
                self.state["user_choice"] = user_input.strip().upper() == "Y"
            else:
                self.state["user_inputs"] = [*self.state["user_inputs"], user_input]
            self.state["history"] = [*self.state["history"], {"role": "user", "content": user_input}]
            return user_input
        return None

    @router(handle_user_input)
    def routing(self) -> str:
        if "completed" in self.state and self.state["completed"]:
            return "event_say_goodbye"
        if len(self.state["user_inputs"]) < 2:
            return "event_request_user_input"
        elif "user_choice" in self.state:
            if self.state["user_choice"]:
                return "event_say_goodbye"
            elif self.state["retry_count"] > 0:
                return "event_retry"
            else:
                return "event_say_goodbye"
        else:
            return "event_start_processing"

    @listen("event_retry")
    def handle_retry(self):
        print(f"Retrying... {self.state["retry_count"]} attempts left")        
        self.state["retry_count"] -= 1
        self.state["attempt_count"] += 1
        self.state["raw_results"] = []

    @listen(or_("event_start_processing", handle_retry))
    def handle_start_processing(self):
        if "raw_results" not in self.state:
            self.state["raw_results"] = []
        for i in range(5):
            self.state["raw_results"].append(f"attempt_{self.state["attempt_count"]}_raw_result_{i+1}")

    @listen(handle_start_processing)
    def handle_processing_complete(self):
        random_result = random.choice(self.state["raw_results"])
        self.state["final_result"] = random_result

    @listen(handle_processing_complete)
    def show_final_result(self):
        output_message = f"Final result: {self.state["final_result"]}\nDo you accept it Y/N?"
        self.__add_output_message(output_message)
        self.state["user_input_type"] = "user_choice"
        print(output_message)
        return "request_user_input"

    @listen("event_request_user_input")
    def handle_request_user_input(self):
        if len(self.state["user_inputs"]) == 0:
            output_message = "Enter something" 
        else:
            output_message = "Enter something else" 
        self.state["user_input_type"] = "user_inputs"
        self.__add_output_message(output_message)
        print(output_message)
        return "request_user_input"

    @listen("event_say_goodbye")
    def handle_say_goodbye(self):
        output_message = ""
        if "completed" in self.state and self.state["completed"]:
            output_message = "This chat already ended. Good bye!"
        elif "user_choice" in self.state and self.state["user_choice"]:
            output_message = "All done. Good bye!"
        else:
            output_message = f"Sorry, you didn't accept the result. I tried {self.state["attempt_count"]} times... Good bye!"
        self.__add_output_message(output_message)
        self.state["completed"] = True
        print(output_message)
        return "event_finished"

    def resume(self, id: str | None = None, user_input: str | None = None) -> tuple[str | None, bool, str | None, list[dict[str, str]]]:
        inputs = {}
        if id and id != self.state["id"]:
            inputs["id"] = id
        if user_input:
            inputs["last_user_input"] = user_input
        output = self.kickoff(inputs=inputs)
        finished = output == "event_finished"
        return self.state["id"], finished, self.state["output_messages"][-1], self.state["history"]

    # def __add_user_input(self, user_input: str):
        

    def __add_output_message(self, message: str):
        self.state["history"] = [*self.state["history"], {"role": "assistant", "content": message}]
        self.state["output_messages"] = [*self.state["output_messages"], message]

    def __pop_from_state(self, key: str, default: Any = None) -> Any:
        if key not in self.state:
            return default
        val = self.state[key]
        del self.state[key]
        return val


def run_chat_loop():
    flow = OutputExampleFlow()
    flow_id = None
    user_input = None
    while True:
        flow_id, _finished, output, _history = flow.resume(id=flow_id, user_input=user_input)
        if _finished:
            break
        if output and len(output) > 0:
            print(f"[Assistant]: {output}")
        user_input = input("[You]: ").strip() 
    flow.plot("my_flow_plot")


def main():
    print("Starting crewai workflow...")
    run_chat_loop()
    print("Crewai workflow completed successfully!")

if __name__ == "__main__":
    # asyncio.run(main())
    main()