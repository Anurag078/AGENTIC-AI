 # Agent loop & task execution logic
from agent.decision import decide_task
from agent.tasks import execute_task

def agent_executor():
    # """
    # Main agent loop:
    # - Takes user input
    # - Decides which task to perform
    # - Executes the task using LLM
    # - Prints the result
    # """
    print("Agent ready!(type 'exit to quit)\n")
    while True:
        user_input= input("You: ")
        if user_input.lower() == "exit":
            print("Agent shutting down. Goodbye!")
            break

        task = decide_task(user_input)

        if task == "echo":
            print("Agent:", user_input)
        else:
            result = execute_task(task, user_input)
            print("Agent:", result)