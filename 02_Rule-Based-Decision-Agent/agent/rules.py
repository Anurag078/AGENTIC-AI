"""
Rule-based decision agent module.
This module defines if-then rules to generate predefined responses for a customer service chatbot.
"""


import re

class RuleBasedAgent:
    def __init__(self):
        self.rules = [
            (["hello", "hi", "greetings"], "Hello! How can I assist you today?"),
            (
                ["orders", "purchase", "buy"],
                "I see you have an inquiry about an order. Could you please provide the order number?",
            ),
            (
                ["complaint", "issue", "problem"],
                "I'm sorry for the inconvenience. Could you tell me more about the issue?",
            ),
            (
                ["refund", "return"],
                "I understand you need help with a refund. Let's work on getting that sorted out.",
            ),
            (
                ["thank", "thanks"],
                "You're welcome! Is there anything else I can help you with?",
            ),
        ]

    def process_input(self, user_input: str) -> str:
        """
        Process the user input with predefined rules.

        Args:
           user_input (str): The input text received from the user.

        Returns:
           str: The chatbot's response based on matched rules.
        """
        text = user_input.lower()
        for keywords, response in self.rules:
            for keyword in keywords:
                # Use regex word boundaries to match whole words only
                if re.search(rf"\b{re.escape(keyword)}\b", text):
                    return response

        return "Could you please provide more details or clarify your request?"
