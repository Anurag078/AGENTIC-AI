import os
from agent.rules import RuleBasedAgent
from agent.state_manager import StateManager
from agent.langchain_deepseek import LangChainGeminiClient


def main():
    print("Welcome to The Rule based Decision Agent with Streaming!")

    state_manager = StateManager()
    rules_agent = RuleBasedAgent()
    gemini_client = LangChainGeminiClient()

    session_id = "session1"
    state_manager.initialize_state(session_id)

    fallback_response = "Could you please provide more details or clarify your request?"

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("GoodBye!")
            break

        # Process using rule-based logic first
        response = rules_agent.process_input(user_input)

        # If rule-based response is fallback, call DeepSeek API with streaming
        if response == fallback_response:
            print("Agent: ", end="", flush=True)  # Start agent response
            messages = [{"role": "user", "content": user_input}]

            # Stream the response token by token
            try:
                full_response = ""
                for chunk in gemini_client.get_completion_streaming_generator(
                    messages
                ):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                print()  # New line after streaming is complete
            except Exception as e:
                print(f"Sorry, I'm having trouble processing your request: {str(e)}")
        else:
            print("Agent:", response)

        #     if "error" in deep_response:
        #         response = "Sorry, I'm having trouble processing your request."
        #     else:
        #         try:
        #             response = deep_response["choices"][0]["message"]["content"]
        #         except (KeyError, IndexError):
        #             response = "Sorry, I couldn't parse the response from the API."

        # print("Agent:", response)


if __name__ == "__main__":
    main()
