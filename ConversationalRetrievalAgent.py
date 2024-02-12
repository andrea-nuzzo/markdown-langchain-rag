from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
import os
load_dotenv()

# Set the OpenAI API key from the environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class ConversationalRetrievalAgent:
    # Initialize the ConversationalRetrievalAgent with a vector database and a temperature for the OpenAI model
    def __init__(self, vectordb, temperature=0.5):
        self.vectordb = vectordb
        self.llm = OpenAI(temperature=temperature)
        self.chat_history = []
    
    # Method to get the chat history as a string    
    def get_chat_history(self, inputs):
        res = []
        for human, ai in inputs:
            res.append(f"Human:{human}\nAI:{ai}")
        return "\n".join(res)
    
    # Method to set up the bot
    def setup_bot(self):
         # Create a retriever from the vector database
        retriever = self.vectordb.as_retriever(search_kwargs={"k": 4})
        # Create a ConversationalRetrievalChain from the OpenAI model and the retriever
        self.bot = ConversationalRetrievalChain.from_llm(
            self.llm, retriever, return_source_documents=True, get_chat_history=self.get_chat_history
        )
    # Method to ask a question to the bot
    def ask_question(self, query):
        # Invoke the bot with the question and the chat history
        result = self.bot.invoke({"question": query, "chat_history": self.chat_history})
         # Append the question and the bot's answer to the chat history
        self.chat_history.append((query, result["answer"]))
         # Return the bot's answer
        return result["answer"]
