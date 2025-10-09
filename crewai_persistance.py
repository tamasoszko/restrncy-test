#!/usr/bin/env python3
"""
Simple in-memory FlowPersistence implementation for CrewAI
"""

import json
import threading
from typing import Any, Dict, Optional
from datetime import datetime

from pydantic import BaseModel
from crewai.flow.persistence.base import FlowPersistence


class InMemoryFlowPersistence(FlowPersistence):
    """
    Simple in-memory implementation of FlowPersistence.
    
    This implementation stores flow states in memory using a dictionary.
    It's thread-safe and suitable for development and testing.
    Note: Data is lost when the process terminates.
    """
    
    def __init__(self):
        """Initialize the in-memory persistence."""
        self._storage: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._initialized = False
    
    def init_db(self) -> None:
        """Initialize the persistence backend.
        
        For in-memory storage, this just marks the instance as initialized.
        """
        with self._lock:
            if not self._initialized:
                self._storage = {}
                self._initialized = True
                print("InMemoryFlowPersistence initialized")
    
    def save_state(
        self, 
        flow_uuid: str, 
        method_name: str, 
        state_data: Dict[str, Any] | BaseModel
    ) -> None:
        """Persist the flow state after method completion.
        
        Args:
            flow_uuid: Unique identifier for the flow instance
            method_name: Name of the method that just completed
            state_data: Current state data (either dict or Pydantic model)
        """
        if not self._initialized:
            self.init_db()
        
        with self._lock:
            # Convert Pydantic model to dict if needed
            if isinstance(state_data, BaseModel):
                state_dict = state_data.model_dump()
            else:
                state_dict = state_data
            
            # Create or update the flow state
            if flow_uuid not in self._storage:
                self._storage[flow_uuid] = {}
            
            # Store the state with metadata
            self._storage[flow_uuid].update({
                'last_method': method_name,
                'last_updated': datetime.now().isoformat(),
                'state': state_dict
            })
            
            print(f"Saved state for flow {flow_uuid} after method {method_name}")
    
    def load_state(self, flow_uuid: str) -> Optional[Dict[str, Any]]:
        """Load the most recent state for a given flow UUID.
        
        Args:
            flow_uuid: Unique identifier for the flow instance
            
        Returns:
            The most recent state as a dictionary, or None if no state exists
        """
        if not self._initialized:
            self.init_db()
        
        with self._lock:
            if flow_uuid in self._storage:
                state_info = self._storage[flow_uuid]
                print(f"Loaded state for flow {flow_uuid} (last method: {state_info.get('last_method', 'unknown')})")
                return state_info.get('state')
            else:
                print(f"No state found for flow {flow_uuid}")
                return None
    


# Example usage and testing
def test_persistence():
    """Test the in-memory persistence implementation."""
    print("Testing InMemoryFlowPersistence...")
    
    # Create persistence instance
    persistence = InMemoryFlowPersistence()
    
    # Initialize
    persistence.init_db()
    
    # Test data
    test_flow_uuid = "test-flow-123"
    test_state = {
        "user_input": "dogs",
        "num_jokes": 3,
        "jokes": ["Why don't dogs make good DJs? Because they have such ruff beats!"],
        "best_joke": "Why don't dogs make good DJs? Because they have such ruff beats!",
        "stage": "complete"
    }
    
    # Test saving state
    persistence.save_state(test_flow_uuid, "generate_jokes", test_state)
    
    # Test loading state
    loaded_state = persistence.load_state(test_flow_uuid)
    print(f"Loaded state: {loaded_state}")
    
    # Test loading again to verify persistence
    loaded_state_2 = persistence.load_state(test_flow_uuid)
    print(f"Loaded state again: {loaded_state_2}")
    
    # Test with different method
    persistence.save_state(test_flow_uuid, "select_best_joke", {"best_joke": "Test joke"})
    final_state = persistence.load_state(test_flow_uuid)
    print(f"Final state after second save: {final_state}")
    
    print("Persistence test completed!")


if __name__ == "__main__":
    test_persistence()
