import os
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv
from ddgs import DDGS

load_dotenv()

def web_search(query: str) -> str:
    """
    Search the web for real-time information, weather, or current events.
    Useful when you need up-to-date data that is not in your training set.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return "No search results found."
            
            formatted_results = []
            for r in results:
                formatted_results.append(f"Title: {r['title']}\nSnippet: {r['body']}\nSource: {r['href']}\n")
            return "\n".join(formatted_results)
    except Exception as e:
        return f"Search error: {str(e)}"

class GeminiClient:
    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash"):
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY")
        
        if not api_key:
            raise ValueError("API key is required. Please set GOOGLE_API_KEY in your .env file.")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model

    def get_completion(self, messages):
        """
        Send a conversation (list of messages) to Google Gemini API using the new SDK.
        Supports automatic tool use for web searching.

        Args:
            messages (list): [{"role": "user"/"assistant"/"system", "content": str}, ...]

        Returns:
            str: The model's response content, or error message.
        """
        try:
            # Get current date and time
            current_date = datetime.now().strftime("%A, %B %d, %Y")
            
            # Convert messages to Gemini SDK format
            gemini_messages = []
            system_instruction = f"Today's date is {current_date}. You have access to a web_search tool for real-time info. "
            
            for msg in messages:
                if msg["role"] == "system":
                    system_instruction += msg["content"]
                else:
                    role = "user" if msg["role"] == "user" else "model"
                    gemini_messages.append({"role": role, "parts": [{"text": msg["content"]}]})

            # Prepend context to the first user message if system_instruction exists
            if gemini_messages and gemini_messages[0]["role"] == "user":
                gemini_messages[0]["parts"][0]["text"] = system_instruction + "\n\n" + gemini_messages[0]["parts"][0]["text"]

            # Generate content with tools
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=gemini_messages,
                config=types.GenerateContentConfig(
                    tools=[web_search]
                )
            )
            
            if response and response.text:
                return response.text.strip()
            else:
                return "No response from Gemini or tool execution failed."
        except Exception as e:
            return f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    client = GeminiClient()
    messages = [
        {"role": "user", "content": "Hello, what can you do?"},
    ]
    print(client.get_completion(messages))