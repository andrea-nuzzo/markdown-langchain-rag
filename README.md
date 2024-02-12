# LangChain Markdown RAG

This project implements a Retrieval-Augmented Generation (RAG) system using the LangChain library. It is designed to work with documents in Markdown format, allowing querying and obtaining relevant information from a collection of documents.

## Prerequisites

Make sure you have Python version 3.10.10 installed on your system. Also, you will need `pip` to install dependencies.

## Installation

To begin, clone this repository on your local system using the following command:

```bash
git clone https://github.com/yourusername/langchain-markdown-rag.git
```

Change to project directory:

```bash
cd langchain-markdown-rag
```

Create a Python virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use  `venv\Scripts\activate`
```

Once the virtual environment is activated, install the project dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Before running the application, you must configure the environment variables:

- Copy the file .env.example into .env:
```bash
cp .env.example .env
```

Open the .env file and enter your OpenAI API keys.

- Insert your markdown files in the project's `markdown_folder` directory. Make sure that all markdown files you wish to analyse are present in this directory before proceeding. 
  
- Edit the `main.py` file to include the specific questions you wish to ask the system. You will find a designated area in the file where you can enter or edit questions.

## Running

After setting up the environment, you can run the project with:

```bash
 python main.py
```

## Project Structure

- **main.py**: The main input file for running the RAG system.
- **DocumentManager.py**: It managed the loading and segmentation of Markdown documents.
- **EmbeddingManager.py**: Responsible for the creation and persistence of embeddings.
- **ConversationalRetrievalAgent.py**: It manages the conversation-based information retrieval system.


