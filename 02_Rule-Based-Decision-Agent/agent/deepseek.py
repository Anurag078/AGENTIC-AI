import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")  # Use uppercase for consistency


class DeepSeekClient:
    def __init__(self, api_key: str = api_key):
        if not api_key:
            raise ValueError(
                "API key is required. Please set API_KEY in your .env file."
            )
        self.api_key = api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def get_completion(self, messages, model: str = "deepseek/deepseek-r1:free"):
        """
        Send a request to the Deepseek R1 API.

        Args:
            messages (list): List of messages in the conversation.
            model (str): The model to be used for response generation.

        Returns:
            dict: API response as a dictionary.
        """
        payload = {"model": model, "messages": messages}
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "details": getattr(e.response, "text", None)}


if __name__ == "__main__":
    client = DeepSeekClient()
    messages = [{"role": "user", "content": "hello, how can I reset my password?"}]
    result = client.get_completion(messages)
    print(result)

# response = requests.post(
#     url="https://openrouter.ai/api/v1/chat/completions",
#     headers={
#         "Authorization": f"Bearer {api_key}",
#     },
#     json={
#         "model": "mistralai/mistral-7b-instruct",
#         "messages":[

#         ]
#     }
# )

# Print response
# print(response.status_code)   # Should be 200 if success
# print(response.json())
