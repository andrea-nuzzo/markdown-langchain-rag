
from DocumentManager import DocumentManager
from EmbeddingManager import EmbeddingManager
from ConversationalRetrievalAgent import ConversationalRetrievalAgent

def main():
    # Initialising and loading documents
    doc_manager = DocumentManager('./marckdown_folder')
    doc_manager.load_documents()
    doc_manager.split_documents()

    # Creation and persistence of embeddings
    embed_manager = EmbeddingManager(doc_manager.all_sections)
    embed_manager.create_and_persist_embeddings()

    # Setup and use of conversation bots
    bot = ConversationalRetrievalAgent(embed_manager.vectordb)
    bot.setup_bot()
    print(bot.ask_question("Question one"))
    print(bot.ask_question("Question two"))
    print(bot.ask_question("Question three"))

if __name__ == "__main__":
    main()