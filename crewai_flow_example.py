#!/usr/bin/env python3
"""
Example of using InMemoryFlowPersistence with CrewAI Flow
"""

import asyncio
from typing import Dict, Any
from pydantic import BaseModel

from crewai.flow import Flow, start
from crewai.flow.persistence import persist
from crewai_persistance import InMemoryFlowPersistence


class JokeFlowState(BaseModel):
    """State for the joke generation flow."""
    topic: str = ""
    num_jokes: int = 3
    jokes: list[str] = []
    best_joke: str = ""
    stage: str = "start"


class JokeFlow(Flow):
    """Example flow that uses in-memory persistence."""
    
    def __init__(self):
        super().__init__()
        self.state = JokeFlowState()
    
    @start()
    @persist(InMemoryFlowPersistence())
    def collect_user_input(self) -> str:
        """Collect user input for joke generation."""
        print("Welcome! I'll help you generate the best joke.")
        
        # Simulate user input collection
        topic = input("What topic would you like a joke about? ").strip()
        if not topic:
            topic = "programming"
        
        try:
            num_jokes = int(input("How many jokes should I generate? (default 3): ").strip() or "3")
        except ValueError:
            num_jokes = 3
        
        # Update state
        self.state.topic = topic
        self.state.num_jokes = num_jokes
        self.state.stage = "generating"
        
        return f"Collected topic: {topic}, number of jokes: {num_jokes}"
    
    @persist(InMemoryFlowPersistence())
    def generate_jokes(self) -> str:
        """Generate jokes about the topic."""
        print(f"Generating {self.state.num_jokes} jokes about {self.state.topic}...")
        
        # Simple joke generation (in real implementation, this would use LLM)
        jokes_db = {
            "dogs": [
                "Why don't dogs make good DJs? Because they have such ruff beats!",
                "What do you call a dog magician? A labracadabrador!",
                "Why did the dog go to the bank? To make a de-paws-it!"
            ],
            "cats": [
                "Why don't cats play poker in the jungle? Too many cheetahs!",
                "What do you call a cat that's been caught in the rain? A soggy pussycat!",
                "Why did the cat break up with the dog? It was a purr-sonal decision!"
            ],
            "programming": [
                "Why do programmers prefer dark mode? Because light attracts bugs!",
                "How many programmers does it take to change a light bulb? None, that's a hardware problem!",
                "Why did the programmer quit his job? He didn't get arrays!"
            ]
        }
        
        import random
        topic_jokes = jokes_db.get(self.state.topic.lower(), jokes_db["programming"])
        
        # Generate the requested number of jokes
        self.state.jokes = random.sample(topic_jokes, min(self.state.num_jokes, len(topic_jokes)))
        self.state.stage = "selecting"
        
        print(f"Generated {len(self.state.jokes)} jokes:")
        for i, joke in enumerate(self.state.jokes, 1):
            print(f"{i}. {joke}")
        
        return f"Generated {len(self.state.jokes)} jokes"
    
    @persist(InMemoryFlowPersistence())
    def select_best_joke(self) -> str:
        """Select the best joke from the generated ones."""
        print("Selecting the best joke...")
        
        # Simple selection (in real implementation, this would use LLM)
        if self.state.jokes:
            self.state.best_joke = self.state.jokes[0]  # Just pick the first one
            self.state.stage = "complete"
            
            print(f"Selected best joke: {self.state.best_joke}")
            return f"Selected: {self.state.best_joke}"
        else:
            return "No jokes to select from"
    
    @persist(InMemoryFlowPersistence())
    def present_result(self) -> str:
        """Present the final result to the user."""
        print("\n" + "="*50)
        print("🎭 FINAL RESULT")
        print("="*50)
        print(f"Topic: {self.state.topic}")
        print(f"Best Joke: {self.state.best_joke}")
        print("="*50)
        
        return f"Presented result for topic: {self.state.topic}"


async def main():
    """Main function to run the flow example."""
    print("Starting CrewAI Flow with InMemoryFlowPersistence...")
    
    # Create and run the flow
    flow = JokeFlow()
    
    try:
        # Run the flow
        result = await flow.kickoff()
        print(f"\nFlow completed with result: {result}")
        
    except KeyboardInterrupt:
        print("\nFlow interrupted by user")
    except Exception as e:
        print(f"Flow error: {e}")
    
    print("Flow example completed!")


if __name__ == "__main__":
    asyncio.run(main())
