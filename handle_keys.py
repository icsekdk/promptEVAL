# import os
# from dotenv import load_dotenv
# import openai
# from anthropic import Anthropic
# import google.generativeai as genai
# import together


# # Load environment variables
# load_dotenv()

# # Configure API clients
# openai.api_key = os.getenv("OPENAI_API_KEY")
# anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# together.api_key = os.getenv("TOGETHER_API_KEY")

import os
from dotenv import load_dotenv
import openai
from anthropic import Anthropic
import google.generativeai as genai

def initialize_apis():
    """Initialize all API clients from environment variables"""
    load_dotenv()
    
    # Initialize OpenAI
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Initialize Anthropic
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Initialize Google
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        genai.configure(api_key=google_api_key)
    return openai_client, anthropic_client

# Initialize global clients
openai_client, anthropic_client = initialize_apis()