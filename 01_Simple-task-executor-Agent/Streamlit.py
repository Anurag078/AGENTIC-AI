import streamlit as st
from agent.decision import decide_task
from agent.tasks import execute_task


st.set_page_config(page_title="AI Task Executor", page_icon="🤖", layout="centered")
st.title("🤖 Simple Task Executor Agent (Gemini-powered)")
st.markdown(
    "Perform summarization, translation, and sentiment analysis using Gemini API."
)

user_input = st.text_area("Enter your text:", height=150)
if st.button("Run Agent"):
    if user_input.strip():
        task = decide_task(user_input)
        if task == "echo":
            st.info(user_input)
        else:
            result = execute_task(task, user_input)
            st.success(result)
    else:
        st.warning("⚠️ Please enter some text before running the agent.")
