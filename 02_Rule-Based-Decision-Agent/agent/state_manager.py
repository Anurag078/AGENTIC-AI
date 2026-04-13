"""
State management module.
This module provides basic state management for a customer service chatbot.
"""

class StateManager:
    def __init__(self):
        # Dictionary to store state keyed by session or user ID.
        self.states = {}

    def initialize_state(self, session_id: str):
        """
        Initialize a new conversation state for a session.

        Args:
            session_id (str): Unique identifier for the conversation session.
        """
        self.states[session_id] = "NEW"

    def update_state(self, session_id: str, new_state: str):
        """
        Update the state of the given session.

        Args:
            session_id (str): Unique identifier for the conversation session.
            new_state (str): New state to be assigned.
        """
        self.states[session_id] = new_state

    def get_state(self, session_id: str) -> str:
        """
        Retrieve the current state of a conversation session.

        Args:
            session_id (str): Unique identifier for the conversation session.

        Returns:
            str: The state of the session, or 'NEW' if the session is not found.
        """
        return self.states.get(session_id, "NEW")

    def reset_state(self, session_id: str):
        """
        Reset the state of the given session to 'NEW'.

        Args:
            session_id (str): Unique identifier for the conversation session.
        """
        self.states[session_id] = "NEW"
