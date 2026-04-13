import streamlit as st
from agent.memory import ConversationMemory
from agent.state_manager import StateManager
from agent.rag import RAGRetriever
from agent.deepseek import GeminiClient

st.set_page_config(page_title="Memory-Enabled Conversational Agent", page_icon="🧠", layout="centered")

st.title("🧠 Memory-Enabled Conversational Agent")
st.markdown("Chat with an agent that remembers context and retrieves relevant information from past conversations.")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = "session1"
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "state_manager" not in st.session_state:
    st.session_state.state_manager = StateManager()
if "rag" not in st.session_state:
    st.session_state.rag = RAGRetriever()
if "llm_client" not in st.session_state:
    st.session_state.llm_client = GeminiClient()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to memory and chat history
    st.session_state.memory.add_message(st.session_state.session_id, "user", user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Add user message to RAG retriever
    st.session_state.rag.add_document(
        doc_id=f"{st.session_state.session_id}_{len(st.session_state.memory.get_history(st.session_state.session_id))}",
        text=user_input
    )

    # Retrieve relevant context and augment prompt
    augmented_prompt = st.session_state.rag.augment_prompt(user_input, top_k=3)

    # Prepare messages for LLM
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    # Add history except the last message
    for msg in st.session_state.memory.get_history(st.session_state.session_id)[:-1]:
        messages.append(msg)
    
    # Add the augmented prompt as the current user message
    messages.append({"role": "user", "content": augmented_prompt})

    # Get response from LLM
    response = st.session_state.llm_client.get_completion(messages)

    # Add assistant response to memory and chat history
    st.session_state.memory.add_message(st.session_state.session_id, "assistant", response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Display assistant response
    st.chat_message("assistant").markdown(response)