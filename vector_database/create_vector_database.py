import chromadb
from argparse import ArgumentParser

from chromadb.config import Settings

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

    # Start the ChromaDB client
    client = chromadb.HttpClient(
        host=args.host_name, 
        port=args.port_number,
        settings=Settings(allow_reset=True)
    )
    client.get_or_create_collection(args.collection_name)

if __name__ == "__main__":
    main()
