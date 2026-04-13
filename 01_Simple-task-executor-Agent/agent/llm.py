import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Configure the new Gemini client (v2 SDK)
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})

def llm_call(prompt, model="gemini-2.5-flash", temperature=0.5):
    # i can call Google Generative AI model with a prompt
    #  Args:
        # prompt (str): Input text for the model
        # model (str): Model name
        # temperature (float): Creativity level (0 = deterministic, 1 = creative)
    
    # Returns
        # str: Model-generated response or error message
    try:
        # Prepend date for context
        full_prompt = f"Today's date is Saturday, April 11, 2026. {prompt}"
        response = client.models.generate_content(
            model=model,
            contents=full_prompt,
            config={"temperature": temperature}
        )

        if response and response.text:
            return response.text.strip()
        else:
            return "No response from Model"
    except Exception as e:
        return f"❌ Error calling LLM: {str(e)}"