# DunderFetch

Welcome to the Scranton Branch's latest innovation in paper-related query processing! This project implements a Question Answering (Q/A) bot for "The Office" (US version) using Retrieval-Augmented Generation (RAG). It's so smart, even a Lackawanna County volunteer sheriff's deputy could use it!

## Features

- Uses SentenceTransformer ('all-MiniLM-L6-v2') for high-quality embeddings
- Employs FAISS for fast and scalable similarity search
- Utilizes Llama 3.1 through Ollama for powerful text generation
- Implements a RAG pipeline for context-aware question answering
- Handles context better than Michael understanding "That's what she said" jokes

## Prerequisites

- Python 3.7+ (no experience selling paper required)
- Ollama installed with Llama 3.1 model available (must be more intelligent than Kevin's famous chili recipe)
- [Data](https://docs.google.com/spreadsheets/d/18wS5AAwOh8QO95RwHLS95POmSNKA2jjzdt0phrxeAE0/edit?gid=747974534#gid=747974534) for the project. [Credits](https://www.reddit.com/r/datasets/comments/b30288/every_line_from_every_episode_of_the_office/)

## Installation

1. Clone this repository faster than Jim can prank Dwight:
   ```
   git clone https://github.com/ironbongjr/DunderFetch.git
   cd DunderFetch
   ```

2. Install the required packages (no Schrute Bucks or Stanley Nickels accepted):
   ```
   pip install sentence-transformers faiss-cpu langchain langchain_community pandas numpy
   ```

3. Ensure Ollama is installed and the Llama 3.1 model is available (and not hidden in the warehouse).

## Usage

1. Run the script (preferably not during a fire drill):
   To process data and create embeddings
   ```
   python process.py
   ```
2. Run the script promt:
   To load llama and ask questions
   ```
   python prompt.py
   ```

3. Ask questions using the `ask_question` function:
   ```python
   ask_question("What was that funny thing someone said in the conference room?")
   ask_question("Who pulled a prank in the office?")
   ```

## Customization

- Adjust the `k` parameter in `get_relevant_documents` to retrieve more or fewer similar documents.
- Modify the prompt template in `prompt_template` to change how the model generates answers.
- Experiment with different SentenceTransformer models, but choose wisely - you don't want your bot to be as unreliable as Andy's anger management

## Limitations

- The current implementation loads all data into memory, which may not be suitable for very large datasets.
- The quality of answers depends on the coverage and quality of the input script data.
- Occasionally generates responses as awkward as a Michael Scott improv scene

## Future Improvements

- Implement a user interface prettier than Pam's watercolors
- Expand to handle multi-turn conversations
- Implement methods to update or add new data to the knowledge base
- Optimize performance to be as smooth as Jim's camera glances

## Contributing

Contributions to improve the bot are welcome. Please feel free to submit a Pull Request (but no Todd Packer-style comments, please).

## Acknowledgments

- Thanks to the creators of SentenceTransformer, FAISS, LangChain, and Ollama for their excellent tools.
- Inspirational credit to the documentary crew who captured all these memorable moments at Dunder Mifflin.

Remember, using this Q/A bot is a lot like working at Dunder Mifflin - it's not just about the paper, it's about the people... and the questions they ask!
