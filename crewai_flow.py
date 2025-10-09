import asyncio
import random


from crewai.flow import or_, router
from crewai.flow.flow import Flow, listen, start
from crewai.flow.persistence.decorators import persist
import openlit

from crewai_event_listener import CrewAiCustomListener
from crewai_persistance import InMemoryFlowPersistence

openlit.init(disable_metrics=True, otlp_endpoint="http://127.0.0.1:4318")

event_listener = CrewAiCustomListener()
persistence = InMemoryFlowPersistence()

# @persist(persistence)  # Using InMemoryFlowPersistence instance
@persist()
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

    @router(initialize)
    def routing(self):
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
        print(f"Final result: {self.state["final_result"]}\nDo you accept it Y/N?")
        return "request_user_accept"

    @listen("event_request_user_input")
    def handle_request_user_input(self):
        return "request_user_input"

    @listen("event_say_goodbye")
    def handle_say_goodbye(self):
        if "user_choice" in self.state and self.state["user_choice"]:
            print("All done. Good bye!")
        else:
            print(f"Sorry, you didn't accept the result. I tried {self.state["attempt_count"]} times... Good bye!")
        return "event_finished"

def start_flow():
    flow = OutputExampleFlow()
    flow.plot("my_flow_plot")
    flow_id = flow.state["id"]
    while True:
        output = flow.kickoff(inputs={"id": flow_id})
        if output == "request_user_input":
            user_input = input("You: ").strip() 
            flow.state["user_inputs"].append(user_input)
            continue
        if output == "request_user_accept":
            user_choice = input("You: ").strip().upper()
            if user_choice == "Y":
                flow.state["user_choice"] = True
            else:
                flow.state["user_choice"] = False
            continue
        if output == "event_finished":
            break
        print(output)



def main():
    print("Starting crewai workflow...")
    start_flow()
    print("Crewai workflow completed successfully!")

if __name__ == "__main__":
    # asyncio.run(main())
    main()