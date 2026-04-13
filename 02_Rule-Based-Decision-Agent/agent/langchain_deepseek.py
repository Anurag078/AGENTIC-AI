import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

class LangChainGeminiClient:
    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash"):
        """
        Standardized Gemini Client using the new Google SDK v2.
        Bypasses LangChain versioning issues to ensure stable v1 API usage.
        """
        # Load API key from environment if not provided
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY")
            
        if not api_key:
            raise ValueError(
                "API key is required. Please set GOOGLE_API_KEY in your .env file."
            )
            
        self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        self.model_name = model

    def get_completion_streaming_generator(self, messages):
        """
        Streams a conversation to Google Gemini and yields text chunks.
        Matches the interface expected by main.py.
        """
        try:
            # Convert messages to Gemini SDK v2 format
            gemini_messages = []
            system_instruction = "Today's date is Saturday, April 11, 2026. "
            
            for msg in messages:
                if msg["role"] == "system":
                    system_instruction += msg["content"]
                elif msg["role"] == "user":
                    gemini_messages.append({"role": "user", "parts": [{"text": msg["content"]}]})
                elif msg["role"] == "assistant":
                    gemini_messages.append({"role": "model", "parts": [{"text": msg["content"]}]})

            # Prepend context to the first user message to avoid system_instruction field errors
            if gemini_messages and gemini_messages[0]["role"] == "user":
                gemini_messages[0]["parts"][0]["text"] = system_instruction + gemini_messages[0]["parts"][0]["text"]

            # Stream the response
            response_iterator = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=gemini_messages,
                config=None
            )

            for chunk in response_iterator:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            yield f"\n[Error calling Gemini API: {str(e)}]"
