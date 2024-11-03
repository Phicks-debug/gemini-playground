# Gemini Playground

This repository contains Python scripts that demonstrate various applications of Google's Gemini AI model. The scripts showcase different functionalities and use cases of the Gemini API.

## Files in the Repository

1. `1_simple_chat.py`: A simple chat application using the Gemini AI model.
2. `2_agent.py`: An AI agent implementation using Gemini for more complex interactions.
3. `3_search_reranking.py`: A script that performs search and reranking using Gemini.
4. `utils.py`: A utility file containing helper functions used across the other scripts.

## Setup and Usage

### Prerequisites

- Python 3.7 or higher
- Google API key for Gemini

### Installation

1. Clone this repository:

   ```bash
   git clone <repository-url>
   cd gemini
   ```

2. Install the required packages:

   ```bash
   pip install google-generativeai python-dotenv termcolor wikipedia
   ```

   or

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your Google API key:

   ```.evn
   GOOGLE_API_KEY=your_api_key_here
   ```

### Running the Scripts

1. Simple Chat (`1_simple_chat.py`):

   ```bash
   python 1_simple_chat.py
   ```

   This script sets up a chat session with the Gemini model. You can interact with the AI by typing your messages.

2. AI Agent (`2_agent.py`):

   ```bash
   python 2_agent.py
   ```

   This script is an extend from the `simple_chat.py` for implementing a more complex AI agent flow that can handle various tasks and prompts. The Agent can do **Code live execution** and **Function calling** with 2 defined functino: *Send email* and *Get weather*.

3. Search and Reranking (`3_search_reranking.py`):

   ```bash
   python 3_search_reranking.py
   ```

   This script demonstrates search functionality and complex reranking statergy using the Gemini model. The script is using the knowledge from **Wikipedia**.

## Features

- Simple chat interactions with Gemini AI
- Complex AI agent capabilities
- Search and reranking functionality
- Utility functions for embedding, Wikipedia searches, and more

## Note

Ensure you have the necessary permissions and comply with Google's terms of service when using the Gemini API.

For more detailed information about each script and its functionalities, please refer to the comments within the individual files.
