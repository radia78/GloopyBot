from dataclasses import dataclass
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHROMA_PATH = "chroma"
BASE_URL = "http://192.168.1.103:11434"
EMB_MODEL_NAME = "snowflake-arctic-embed:22m"
CHAT_MODEL_NAME = "phi3:mini"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based only on the above context: {question}
"""

def main() -> None:
    # Prepare the DB
    db = Chroma(
        collection_name = "LOCAL_DOCUMENT_VECTOR_DATABASE", # Database Name
        embedding_function = OllamaEmbeddings(base_url = BASE_URL, model=EMB_MODEL_NAME), # Embedding Model
        collection_metadata = {"hnsw:space": "cosine"} # Distance Metric: Cosine Distance
    )

    # Prepare text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 500,
        length_function = len,
        add_start_index = True
    )

    # Prepare the model
    model = ChatOllama(base_url = BASE_URL, model = CHAT_MODEL_NAME)

    # Ready the prompt template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # Always on loop unless pressed ctrl+c
    while True:
        # Input for query and document(s) directory
        context_text = ""
        query_text = input("Chat prompt: ")
        doc_dir = input("Please input additional document(s) directory, otherwise enter 'none': ")

        # Case if no directory is given
        if doc_dir == "none":
            prompt = prompt_template.format(context=context_text, question=query_text)
            sources = "N/A"

        # Case if a directory is given
        else:
            # Load the document and upload it to the vector database
            loader = DirectoryLoader(doc_dir, glob="*.md")
            chunks = text_splitter.split_documents(loader.load())
            print("Adding document(s) to the vector database.")
            db.add_documents(chunks)

            # Find the relevant results from the query
            results = db.similarity_search_with_relevance_scores(query_text, k=3)
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            sources = [doc.metadata.get("source", None) for doc, _score in results]

        # Format the query to include contexts
        print("Formatting prompt.")
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Get response from LLM
        print("Generating response.")
        response_text = model.invoke(prompt)

        # List the sources and append it to response
        print("Formating response.")
        formatted_response = f"Response: {response_text.content}\nSources: {sources}"
        print(formatted_response)

if __name__ == "__main__":
    main()