import os
import streamlit as st
from agent.rules import RuleBasedAgent
from agent.state_manager import StateManager
from agent.langchain_deepseek import LangChainDeepSeekClient
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")


def main():
    st.title("🤖 Rule-Based Decision Agent with Streaming")

    # Initialize components
    if "rules_agent" not in st.session_state:
        st.session_state.rules_agent = RuleBasedAgent()
    if "state_manager" not in st.session_state:
        st.session_state.state_manager = StateManager()
    if "deepseek_client" not in st.session_state:
        st.session_state.deepseek_client = LangChainDeepSeekClient(api_key)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("How can I help you today?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        fallback_response = (
            "Could you please provide more details or clarify your request?"
        )
        response = st.session_state.rules_agent.process_input(prompt)

        with st.chat_message("assistant"):
            if response == fallback_response:
                # Stream response from API
                message_placeholder = st.empty()
                full_response = ""

                messages = [{"role": "user", "content": prompt}]

                try:
                    for chunk in st.session_state.deepseek_client.get_completion_streaming_generator(
                        messages
                    ):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")

                    message_placeholder.markdown(full_response)
                    response = full_response

                except Exception as e:
                    error_msg = (
                        f"Sorry, I'm having trouble processing your request: {str(e)}"
                    )
                    message_placeholder.markdown(error_msg)
                    response = error_msg
            else:
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
