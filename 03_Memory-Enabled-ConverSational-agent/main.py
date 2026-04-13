from agent.memory import ConversationMemory
from agent.state_manager import StateManager
from agent.rag import RAGRetriever
from agent.deepseek import GeminiClient

def main():
    print("Welcome to the Memory-Enabled Conversational Agent!")
    session_id = "session1"

    # Initialize components
    memory = ConversationMemory()
    state_manager = StateManager()
    rag = RAGRetriever()
    llm_client = GeminiClient()

    state_manager.initialize_state(session_id)

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "bye"]:
            print("GoodBye!")
            break

        memory.add_message(session_id, "user", user_input)

         # Add user message to RAG retriever (for demo, store all messages)
        rag.add_document(doc_id=f"{session_id}_{len(memory.get_history(session_id))}", text=user_input)

        # Retrieve relevant context from memory using RAG
        augmented_prompt = rag.augment_prompt(user_input, top_k=3)

        # prepare messages for llm
        messages = [{"role":"system", "content": "You are a helpful assistant."}]
        for msg in memory.get_history(session_id)[:-1]:
            messages.append(msg)
        
        # Add the latest user message as the augmented prompt
        messages.append({"role": "user", "content": augmented_prompt})
        
        # Get response from LLM
        response = llm_client.get_completion(messages)

        # Store assistant response in memory
        memory.add_message(session_id, "assistant", response)

        # Update state (example: last response)
        state_manager.update_state(session_id, "last_response", response)

        print("Agent:", response)

if __name__ == "__main__":
    main()