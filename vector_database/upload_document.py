import chromadb
from chromadb.config import Settings
from argparse import ArgumentParser
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

EMB_MODEL_NAME = "snowflake-arctic-embed:22m"

def main() -> None:
    parser = ArgumentParser(description="Upload a legal document to the vector database")
    parser.add_argument(
        "-c", 
        "--collection_name", 
        help="Name of the collection. Make sure it's all lowercase and connected by underline.",
        type=str
    )
    parser.add_argument(
        "-d",
        "--doc_dir",
        help="Directory of the legal document(s)",
        type=str
    )
    parser.add_argument(
        "-i", 
        "--host_name", 
        help="Name of the host machine for the vector database.",
        type=str
    )
    parser.add_argument(
        "-p", 
        "--port_number", 
        help="The exposed port for the vector database",
        type=int,
        default=8000
    )
    args = parser.parse_args()

    # Prepare text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 500,
        length_function = len,
        add_start_index = True
    )

    # Start the ChromaDB client
    client = chromadb.HttpClient(
        host=args.host_name, 
        port=args.port_number,
        settings=Settings(allow_reset=True)
    )
    client.get_or_create_collection(args.collection_name)

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