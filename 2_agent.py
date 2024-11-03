import os
import google.generativeai as genai

from utils import process_tool
from dotenv import load_dotenv
from termcolor import cprint
from google.api_core.exceptions import ResourceExhausted
from datetime import datetime as dt
from google.api_core import retry_async

load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))


# THE TEST PROMPT, YOU COULD COPY PASTE TO SEE THE RESULT IN ACtION
test_prompt_1 = "Can you send the email to my boss to infor him about the current weather of Hanoi and HoChiMinh city and congratualation him on his new promotion. Be professional."
test_prompt_2 = "I have the function x2^10 - x8^2 + x^3 * log(20) + C = 100. Solve x using the bisection method. Assuming C is in range (1, 10). Run the code and give me the answer and graph the result into line with different C in range."
test_prompt_3 = """Run a simulation of the Monty Hall Problem with 1,000 trials.

Here's how this works as a reminder. In the Monty Hall Problem, you're on a game
show with three doors. Behind one is a car, and behind the others are goats. You
pick a door. The host, who knows what's behind the doors, opens a different door
to reveal a goat. Should you switch to the remaining unopened door?

The answer has always been a little difficult for me to understand when people
solve it with math - so please run a simulation with Python to show me what the
best strategy is.

Thank you!
"""


system_prompt = f"""Your name is David. You are a helpful assistant.
Here is the current time and date {dt.now().strftime("%Y-%m-%d %H:%M:%S")}
DO NOT call yourself Gemnini or anything related to Google.
Don't make assumptions on the weather. Always use a get_weather tool for the current weather states.
Ask clarifying questions if not enough information is available to complete the request.
My email is: an.tq@techxcorp.com, my Name is Phicks
My boss email is: duy.doan@techxcorp.com, my boss name is Duy Scientist
"""
send_email_func = genai.protos.FunctionDeclaration(
    name="send_email",
    description="Send the email",
    parameters=genai.protos.Schema(
        type="OBJECT",
        properties={
            "sender_email": genai.protos.Schema(
                type="STRING",
                description="The sender email"
            ),
            "recipent_email": genai.protos.Schema(
                type="STRING",
                description="The recipent email"
            ),
            "subject": genai.protos.Schema(
                type="STRING",
                description="The subject of the email"
            ),
            "body": genai.protos.Schema(
                type="STRING",
                description="The body of the email"
            )
        },
        required=[
            "sender_email",
            "recipent_email",
            "subject",
            "body"
        ]
    )
)
get_weather_func = genai.protos.FunctionDeclaration(
    name="get_weather",
    description="Get the current weather state of the city",
    parameters=genai.protos.Schema(
        type="OBJECT",
        properties={
            "city": genai.protos.Schema(
                type="STRING",
                description="The city"
            )
        },
        required=[
            "city"
        ]
    )
)
mode = input("Select agent mode: (1.Function calling), (2.Code agent), (3.Grounding Truth): ")
if mode == "1":
    tools = genai.protos.Tool(function_declarations=[send_email_func, get_weather_func])
elif mode == "2":
    tools = genai.protos.Tool(code_execution=genai.protos.CodeExecution())
else:
    tools = genai.protos.Tool(google_search_retrieval=genai.protos.GoogleSearchRetrieval(
        dynamic_retrieval_config=genai.protos.DynamicRetrievalConfig(
            mode=genai.protos.DynamicRetrievalConfig.Mode.MODE_DYNAMIC,
            dynamic_threshold=0.3
    )))
config = genai.GenerationConfig(
    temperature=0, 
    top_p=1,
    top_k=40,
    # candidate_count=1,
    max_output_tokens=2048
)
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
    generation_config=config,
    tools=tools,
    system_instruction=system_prompt,
    safety_settings={
        'HATE': 'BLOCK_NONE',
        'HARASSMENT': 'BLOCK_NONE',
        'SEXUAL' : 'BLOCK_NONE',
        'DANGEROUS' : 'BLOCK_NONE'
    }
)
max_retry = 3


async def main():
    history = []
    attempt = 0
    
    # Loop for the user insert program
    while True:
        
        user_input = input("You: ")

        # Exit
        if user_input.lower() == "/bye":
            break
        
        # Add into chat history
        history.append(genai.protos.Content(
            parts=[genai.protos.Part(text=user_input)],
            role="user"
        ))
        
        # Loop for the agent to process
        while True:
            
            end_turn = False
            fn_list = []
            full_res = ""

            try:
                # Parse and process the response
                async for chunk in await model.generate_content_async(
                    contents = history,
                    stream = True,
                    request_options={
                        "retry": retry_async.AsyncRetry(predicate=retry_async.if_transient_error),
                        "timeout": 900
                    }   # Retried option if fail
                ):
                    # print(chunk)
                    # If stop reason
                    for candidate in chunk._result.candidates:                    
                        for part in candidate.content.parts:
                            
                            # Add the tool_request into the chat history
                            if fn := part.function_call:
                                history.append(genai.protos.Content(parts=[genai.protos.Part(function_call=fn)],role="model"))
                                fn_list.append(fn)
                                    
                            # Print out the text response
                            if token := part.text:
                                full_res += token
                                cprint(token, "green", end="", flush=True)
                                
                            # Print out the executable_code
                            if code := part.executable_code:
                                cprint(f"\n{code.language.name}", "light_cyan", flush=True)
                                cprint("_"*80)
                                cprint(f"{code.code}", "light_green", end="", flush=True)
                                cprint("_"*80)
                                history.append(genai.protos.Content(parts = [genai.protos.Part(executable_code=code)],role = "model"))
                            
                            # Print out the code execution result
                            if code := part.code_execution_result:
                                cprint(f"{code.outcome.name}", "light_cyan", flush=True)
                                cprint(f"{code.output}\n", "light_yellow", end="", flush=True)
                                history[-1].parts.append(genai.protos.Part(code_execution_result = code))
                                
                            
                            # Check is it is done streaming and "does have some response back"
                            # Because the API raise Error if there is an empty response from the model.
                            if candidate.finish_reason and full_res:
                                
                                # If placeholder had been used means have function, -> possible process tool call
                                if history[-1].role == "model":
                                    history[-1].parts.append(genai.protos.Part(text=full_res))
                                    if not fn_list:
                                        cprint(f"{candidate.finish_reason.name}", "light_red")
                                        end_turn = True
                                        break   # If the model does not have any tool to call, break
                                else:
                                    history.append(genai.protos.Content(parts=[genai.protos.Part(text=full_res)],role="model"))
                                    cprint(f"{candidate.finish_reason.name}", "light_red")
                                    end_turn = True
                                    break
                            # print(history)
                    if end_turn:
                        break
                if end_turn:
                    break
                
                history = await process_tool(fn_list, history)
            except ResourceExhausted:
                    if attempt >= max_retry:
                        raise Exception("Max retry reached")
                    attempt  += 1
                    print(f"Sever overload: retrying {attempt}/{max_retry} ...")
                    await asyncio.sleep(1)
          

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())