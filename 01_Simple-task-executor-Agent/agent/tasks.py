from agent.llm import llm_call
# Tasks
TASKS = {
    "summarize": "Summarize the following text in a concise way:",
    "translate": "Translate the following text to Hindi:",
    "sentiment": "Analyze the sentiment of the following text (Positive, Negative, Neutral):",
    "general_qa": "",
}
def execute_task(task_name, user_input):
    #  """
    # Executes a predefined task using LLM.

    # Args:
    #     task_name (str): The task identifier (summarize, translate, sentiment)
    #     user_input (str): The text provided by user

    # Returns:
    #     str: The LLM-generated output
    # """
    if task_name not in TASKS:
        return f"⚠️ Unknown task: {task_name}" 
    
    prompt = f"{TASKS[task_name]} \n\n{user_input}"
    result = llm_call(prompt)
    return result