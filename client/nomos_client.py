from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from argparse import ArgumentParser
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate

EMB_MODEL_NAME = "snowflake-arctic-embed:22m"
CHAT_MODEL_NAME = "phi3:mini"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based only on the above context: {question}
"""

def main() -> None:
    # Parse the command line arguments
    parser = ArgumentParser(description="Perform inference on legal documents")
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
    parser.add_argument(
        "-e", 
        "--emb_url", 
        help="The url for the Ollama server",
        type=str
    )
    args = parser.parse_args()

    # Start the Chroma client
    client = chromadb.HttpClient(
        host=args.host_name, 
        port=args.port_number,
        settings=Settings(allow_reset=True)
    )
    client.get_or_create_collection(args.collection_name)

    # Prepare the DB
    db = Chroma(
        client = client,
        collection_name = args.collection_name,
        embedding_function = OllamaEmbeddings(base_url=args.emb_url, model=EMB_MODEL_NAME),
        collection_metadata = {"hnsw:space": "cosine"} # Distance Metric: Cosine Distance
    )

    # Prepare the model
    model = ChatOllama(
        base_url=args.emb_url, 
        model = CHAT_MODEL_NAME
    )

    # Ready the prompt template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # Always on loop unless pressed ctrl+c
    while True:
        # Input for query and document(s) directory
        context_text = ""
        query_text = input("Chat prompt: ")

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
        print("Formating response.\n")
        formatted_response = f"Response: {response_text.content}\nSources: {sources}"
        print(formatted_response.join("\n"))

if __name__ == "__main__":
    main()