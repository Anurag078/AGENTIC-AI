class StateManager:
    """
    Manages session or conversation state for each user.
    """

    def __init__(self):
        # Dictionary to store state per session/user
        self.states = {}

    def initialize_state(self, session_id):
        """
        Initialize a new state for a session.

        Args:
            session_id (str): Unique session or user ID.
        """
        self.states[session_id] = {"status": "NEW"}

    def update_state(self, session_id, key, value):
        """
        Update a specific state value for a session.

        Args:
            session_id (str): Unique session or user ID.
            key (str): State key to update.
            value: New value for the state key.
        """
        if session_id not in self.states:
            self.initialize_state(session_id)
        self.states[session_id][key] = value

    def get_state(self, session_id):
        """
        Retrieve the current state dictionary for a session.

        Args:
            session_id (str): Unique session or user ID.

        Returns:
            dict: State dictionary for the session.
        """
        return self.states.get(session_id, {"status": "NEW"})

    def reset_state(self, session_id):
        """
        Reset the state for a session to default.

        Args:
            session_id (str): Unique session or user ID.
        """
        self.states[session_id] = {"status": "NEW"}

# Example usage
if __name__ == "__main__":
    sm = StateManager()
    sm.initialize_state("session1")
    sm.update_state("session1", "last_intent", "ask_about_agentic_ai")
    print(sm.get_state("session1"))
    sm.reset_state("session1")