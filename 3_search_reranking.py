import os
import asyncio
import google.generativeai as genai

from utils import process_tool
from dotenv import load_dotenv

from termcolor import cprint
from datetime import datetime as dt
from google.api_core import retry_async
from google.api_core.exceptions import ResourceExhausted


load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
is_reranking = True

# THE TEST PROMPT OR YOU COULD USE OUR OWN, YOU COULD COPY PASTE TO SEE THE RESULT IN ACtION
test_prompt_1 = "Explain how deep-sea life survives, use the tool provided, I want to know more about the horrendous, urgly face fish."


system_prompt = f"""Your name is David. You are a helpful assistant.
Here is the current time and date {dt.now().strftime("%Y-%m-%d %H:%M:%S")}
You knowledge cut off at December 2023.
DO NOT call yourself Gemnini or anything related to Google.
Ask clarifying questions if not enough information is available to complete the request.

You have access to the Wikipedia API which you will be using to answer a user's query. 
Your job is to generate a list of search queries which might answer a user's question. 
Be creative by using various key-phrases from the user's query. To generate variety of queries, ask questions which are related to  the user's query that might help to find the answer. 
The more queries you generate the better are the odds of you finding the correct answer.

Here is an example:
<example>
user: Tell me about Cricket World cup 2023 winners.

function_call: wikipedia_search(['What is the name of the team that
won the Cricket World Cup 2023?', 'Who was the captain of the Cricket World Cup
2023 winning team?', 'Which country hosted the Cricket World Cup 2023?', 'What
was the venue of the Cricket World Cup 2023 final match?', 'Cricket World cup 2023',
'Who lifted the Cricket World Cup 2023 trophy?'])
</example>
"""
summarize_prompt = """Summarize the latest user question from the chat history into maximum 2 sentence. 
Only return the summary. Your summary question should feel like it has been written by a human.

Here is the chat history:
{}
"""
hypothetical_prompt = """Generate a hypothetical answer to the user's query by using your own knowledge. 
Assume that you know everything about the said topic. 
Do not use factual information, instead use placeholders to complete your answer. 
Your answer should feel like it has been written by a human.

Query: {}"""

wikipedia_search_func = genai.protos.Tool(
    function_declarations = [
        genai.protos.FunctionDeclaration(
            name = "wikipedia_search",
            description = "Search wikipedia for each query and summarize relevant docs.",
            parameters = genai.protos.Schema(
                type="OBJECT",
                properties={
                    "search_queries": genai.protos.Schema(
                        type="ARRAY",
                        description="list of topics that your want to search for",
                        max_items = 5,
                        min_items = 2,
                        items=genai.protos.Schema(
                            type="STRING",
                            description="topics that your want to search for"
                        )
                    )
                },
                required=["search_queries"]
            )
            
        )
    ]
)

config = genai.GenerationConfig(
    temperature=1, 
    top_p=1,
    top_k=40,
    # candidate_count=1,
    max_output_tokens=4096
)
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
    generation_config=config,
    tools=[wikipedia_search_func],
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
                
                if is_reranking:
                    # Get the summarization on the question. (In production, this should run paralle in background)
                    sum_query_model = genai.GenerativeModel('gemini-1.5-flash')
                    response = await sum_query_model.generate_content_async(
                        summarize_prompt.format(history.pop()),
                        generation_config=genai.GenerationConfig(
                            temperature=0,
                            max_output_tokens=512
                        )
                    )
                    cprint(f"Summarized question prompt:\n{response.text}", "grey")
                    
                    # Get the hypothetical answer based on the generated question. (In production, this should run paralle in background)
                    hypothetical_ans_model = genai.GenerativeModel('gemini-1.5-flash')
                    res = await hypothetical_ans_model.generate_content_async(
                        hypothetical_prompt.format(response.text),
                    )
                    cprint(f"Hypothetical answer:\n{res.text}", "grey")
                    top_k = 5
                    query = res.text
                else:
                    query = None
                    top_k = None
                    
                # Pass the summarized question and the top k.
                history = await process_tool(fn_list, history, is_reranking, query, top_k)
                    
            except ResourceExhausted:
                    if attempt >= max_retry:
                        raise Exception("Max retry reached")
                    attempt  += 1
                    print(f"Sever overload: retrying {attempt}/{max_retry} ...")
                    await asyncio.sleep(1)

if __name__ == "__main__":
    mode = input("Reranking search? (y/n):")
    if mode.lower() == 'n':
        is_reranking = False
    asyncio.run(main())