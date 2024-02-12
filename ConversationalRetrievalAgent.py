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
            self.llm,
            retriever,
            return_source_documents=True,
            get_chat_history=self.get_chat_history,
        )

    def generate_prompt(self, question):
        if not self.chat_history:
            # Se è la prima domanda, usa un template specifico senza contesto di conversazione precedente
            prompt = f"You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. \nQuestion: {question}\nContext: \nAnswer:"
        else:
            # If it is the first question, use a specific template without previous conversation context
            context_entries = [f"Question: {q}\nAnswer: {a}" for q, a in self.chat_history[-3:]]
            context = "\n\n".join(context_entries)
            prompt = f"Using the context provided by recent conversations, answer the new question in a concise and informative. Limit your answer to a maximum of three sentences.\n\nContext of recent conversations:\n{context}\n\nNew question: {question}\n\Answer:"
        
        return prompt
    
    # Method to ask a question to the bot
    def ask_question(self, query):
        prompt = self.generate_prompt(query)
        # Invoke the bot with the question and the chat history
        result = self.bot.invoke({"question": prompt, "chat_history": self.chat_history})
        # Append the question and the bot's answer to the chat history
        self.chat_history.append((query, result["answer"]))
        
        # Return the bot's answer
        return result["answer"]
