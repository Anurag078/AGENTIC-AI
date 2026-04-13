class ConversationMemory:
    """
    Stores and retrieves conversation history for a session/user.
    """

    def __init__(self):
        # Dictionary to store history per session/user
        self.histories = {}

    def add_message(self, session_id, role, content):
        """
        Add a message to the conversation history.

        Args:
            session_id (str): Unique session or user ID.
            role (str): 'user', 'assistant', or 'system'.
            content (str): Message text.
        """
        if session_id not in self.histories:
            self.histories[session_id] = []
        self.histories[session_id].append({"role": role, "content": content})

    def get_history(self, session_id):
        """
        Retrieve the full conversation history for a session.

        Args:
            session_id (str): Unique session or user ID.

        Returns:
            list: List of message dicts.
        """
        return self.histories.get(session_id, [])

    def clear_history(self, session_id):
        """
        Clear the conversation history for a session.

        Args:
            session_id (str): Unique session or user ID.
        """
        self.histories[session_id] = []


# Example usage
if __name__ == "__main__":
    memory = ConversationMemory()
    memory.add_message("session1", "user", "Hello!")
    memory.add_message("session1", "assistant", "Hi, how can I help you?")
    print(memory.get_history("session1"))
