import chromadb
from argparse import ArgumentParser
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from chromadb.config import Settings

EMB_MODEL_URL = ""
EMB_MODEL_NAME = "snowflake-arctic-embed:22m"

def main() -> None:
    # Arguments for the script
    parser = ArgumentParser(description="Create vector database for legal documents")
    parser.add_argument(
        "-c", 
        "--collection_name", 
        help="Name of the collection. Make sure it's all lowercase and connected by underline.",
        type=str
    )
    parser.add_argument(
        "-d",
        "--doc_dir",
        help="Directory where document is hosted.",
        type=str 
    )
    args = parser.parse_args()

    # Start the ChromaDB client
    client = chromadb.HttpClient(settings=Settings(allow_reset=True))
    client.create_collection(args.collection_name, )

    # Prepare text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 500,
        length_function = len,
        add_start_index = True
    )

    # Pass the ChromaDB client to LanChain
    db = Chroma(
        client = client,
        collection_name = args.collection_name,
        embedding_function = OllamaEmbeddings(base_url=EMB_MODEL_URL, model=EMB_MODEL_NAME)
    )

    # Load the document and upload it to the vector database
    loader = DirectoryLoader(args.doc_dir, glob="*.md")
    chunks = text_splitter.split_documents(loader.load())
    print("Adding document(s) to the vector database.")
    db.add_documents(chunks)

if __name__ == "__main__":
    main()
