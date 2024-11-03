import os
import google.generativeai as genai

from dotenv import load_dotenv
from utils import get_all_model
from termcolor import cprint

# Load environment variable
load_dotenv()

# Set up API
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# get_all_model()

chat_session = genai.ChatSession(
    model = genai.GenerativeModel("gemini-1.5-flash-002"),
    history = [],
    # enable_automatic_function_calling=True
)
config = genai.GenerationConfig(
    temperature=0.5, 
    top_p=0.8,
    top_k=40,
    candidate_count=1,
    max_output_tokens=2048
)


while True:
    
    user_input = input("You: ")

    # Exit
    if user_input.lower() == "/bye":
        break
    
    # Send request
    response = chat_session.send_message(
        content = user_input,
        generation_config=config,
        stream = True,
    )

    # Parse and print response
    for chunk in response:
        for candidate in chunk._result.candidates:
            for part in candidate.content.parts:
                cprint(part.text, "green", end="", flush=True)

