import textwrap
import wikipedia
import random

import google.generativeai as genai
import asyncio
import numpy as np

from termcolor import cprint
from typing import List
from wikipedia.exceptions import DisambiguationError, PageError



def get_all_model():
    print("List of models that support generateContent:\n")
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(m.name)
    print("List of models that support embedContent:\n")
    for m in genai.list_models():
        if "embedContent" in m.supported_generation_methods:
            print(m.name)
            

async def get_embeddings(content: list[str]) -> np.ndarray:
    embeddings = await genai.embed_content_async('models/text-embedding-004', content, 'SEMANTIC_SIMILARITY')
    embds = embeddings.get('embedding', None)
    embds = np.array(embds).reshape(len(embds), -1)
    return embds


async def dot_product(a: np.ndarray, b: np.ndarray):
    return (a @ b.T)

            
async def process_tool(
    fn_list: List[genai.protos.FunctionCall], 
    history: List, 
    is_reranking: bool = False,
    query: str = None,
    k: int = None
) -> List[genai.protos.Content]:
    
    # Check only if is_reranking set to True then this function is accept query
    if is_reranking and not query:
        raise ValueError("query is required for reranking")
    elif query and not is_reranking:
        raise ValueError("reranking need to be set to accept query")
    elif is_reranking and not k:
        raise ValueError("n is required for reranking")
    elif k and not is_reranking:
        raise ValueError("reranking need to be set to accept n")
    
    fn_results = []
    for fn in fn_list:
        try:
            func = globals().get(fn.name)   # Use globals() to dynamically retrieve and call the function
            if func:
                args = {key: val for key, val in fn.args.items()}
                cprint("_"*80, flush=True)
                cprint(f"using {fn.name} tool", "light_cyan", flush=True)
                result = await func(**args)
                if is_reranking:
                    
                    # Get embeddings for user's query and search results
                    search_res = await get_embeddings(result)
                    embedded_query = await get_embeddings([query])
                    
                    # Calculate similarity score
                    sim_values = (await dot_product(search_res, embedded_query)).flatten()
                    
                    # Sort results by similarity and select top `n`
                    top_indices = np.argsort(sim_values)[-k:][::-1]
                    top_results = [result[i] for i in top_indices]
                    
                    # Combine top results into a single response string
                    result = "\n\n".join(top_results)
                else:
                    result = "\n\n".join(result)
                    
                cprint(result, "light_magenta", flush=True)
                cprint("_"*80, flush=True)
                is_error = False
            else:
                result = f"Function {fn.name} does not found."
                is_error = True
        except Exception as e:
            result = f"Error processing function {fn.name}: {str(e)}"
            is_error = True
        fn_results.append(
            genai.protos.Part(
                function_response=genai.protos.FunctionResponse(
                    name=fn.name, response={"result": result, "is_error": is_error}
                )
            )
        )
    history.append(genai.protos.Content(parts=fn_results, role="user"))
    return history


async def wikipedia_search(search_queries: list[str]) -> list[str]:
    """Search wikipedia for each query and summarize relevant docs."""
    n_topics=3
    search_history = set() # tracking search history
    search_urls = []
    mining_model = genai.GenerativeModel(
        'gemini-1.5-flash',
        safety_settings={
            'HATE': 'BLOCK_NONE',
            'HARASSMENT': 'BLOCK_NONE',
            'SEXUAL' : 'BLOCK_NONE',
            'DANGEROUS' : 'BLOCK_NONE'
        }
    )
    summary_results = []
    
    async def search_wikipedia(query: str):
        # Run the synchronous `wikipedia.search` in an executor to avoid blocking.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, wikipedia.search, query)

    for query in search_queries:
        cprint(f'Searching for "{query}"', "light_yellow")
        search_terms = await search_wikipedia(query)

        cprint(f"Related search terms: {search_terms[:n_topics]}", "light_yellow")
        for search_term in search_terms[:n_topics]: # select first `n_topics` candidates
            if search_term in search_history: # check if the topic is already covered
                continue

            cprint(f'Fetching page: "{search_term}"', "light_yellow")
            search_history.add(search_term) # add to search history

            try:
                # extract the relevant data by using `gemini-1.5-flash` model
                page = await asyncio.to_thread(wikipedia.page, search_term, auto_suggest=False)
                url = page.url
                cprint(f"Information Source: {url}", "light_blue")
                search_urls.append(url)
                page = page.content
                
                response = await mining_model.generate_content_async(textwrap.dedent(f"""\
                    Extract relevant information
                    about user's query: {query}
                    From this source:

                    {page}

                    Note: Do not summarize. Only Extract and return the relevant information
                """))

                urls = [url]
                if response.candidates[0].citation_metadata:
                    extra_citations = response.candidates[0].citation_metadata.citation_sources
                    extra_urls = [source.url for source in extra_citations]
                    urls.extend(extra_urls)
                    search_urls.extend(extra_urls)
                    cprint("Additional citations: " + response.candidates[0].citation_metadata.citation_sources, "light_grey")
                try:
                    text = response.text
                except ValueError as e:
                    cprint(f"Problems with parsing response: {e}", "red")
                    pass
                else:
                    summary_results.append(text + "\n\nBased on:\n  " + ',\n  '.join(urls) + "\n\n")

            except DisambiguationError:
                cprint(f"""Results when searching for "{search_term}" (originally for "{query}")
                were ambiguous, hence skipping""", "light_red")
                continue

            except PageError:
                cprint(f'{search_term} did not match with any page id, hence skipping.', "light_red")
                continue
                
            except:
                cprint(f'{search_term} did not match with any page id, hence skipping.', "light_red")
                continue

    cprint(f"Information Sources:", "light_blue")
    for url in search_urls:
        cprint("   "+url, "light_green")

    """For returning all the search result"""
    # result = "".join(summary_results)
    # cprint(f"Result: {result}", "light_grey")
    # return result
    
    """For returning for reranking"""
    return summary_results


async def send_email(
    sender_email: str, 
    recipent_email: str, 
    subject: str, 
    body: str
):
    """Send the email"""
    return f"""The email have been sent from {sender_email} to {recipent_email} with:
    {subject}
    {body}"""
    

async def get_weather(
    city: str
):
    """Get the current weather state of the city"""
    return f"""{random.choice([
        "sunny",
        "cloudy",
        "rainy",
        "snowy",
        "windy"
    ])} with {random.randint(20, 40)} degrees in {city}"""
