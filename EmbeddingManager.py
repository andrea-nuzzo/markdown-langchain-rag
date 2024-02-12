from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

class EmbeddingManager:
    def __init__(self, all_sections, persist_directory='db'):
        self.all_sections = all_sections
        self.persist_directory = persist_directory
        self.vectordb = None
        
    # Method to create and persist embeddings
    def create_and_persist_embeddings(self):
        # Creating an instance of OpenAIEmbeddings
        embedding = OpenAIEmbeddings()
        # Creating an instance of Chroma with the sections and the embeddings
        self.vectordb = Chroma.from_documents(documents=self.all_sections, embedding=embedding, persist_directory=self.persist_directory)
         # Persisting the embeddings
        self.vectordb.persist()