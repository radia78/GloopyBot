from fastapi import FastAPI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from utils import *

GEN_MODEL_PATH = "models/llm_model/qwen/qwen1.5-1.8b-q4_k_m.gguf"
EMB_MODEL_PATH = "models/emb_model/snowflake-arctic-embed-l-q4_k_m.gguf"
CHROMA_PATH = "chroma"
DOCUMENT_PATH = "chroma/data/short_paper.pdf"

# Initialize the LLM MODELS
callback_manager = CallbackManager([StreamingStdOutCallbackHandler])

# Initialize the generator LLM model (LLAMA 8B)
gen_model = LlamaCpp(
    model_path=GEN_MODEL_PATH,
    temperature=0.75,
    max_tokens=2048,
    top_p=1,
    n_ctx=1048,
    callback_manager=callback_manager,
    verbose=True
)

# Initialize the embedding model for the vector database
emb_model = LlamaCppEmbeddings(model_path=EMB_MODEL_PATH)

# Initialize the pdf loader
pdf_loader = PyPDFLoader(file_path=DOCUMENT_PATH)
pages = pdf_loader.load_and_split()

# Initialize the database
db = Chroma.from_documents(
    pages,
    embedding=emb_model,
    collection_name="LOCAL_DATABASE",
    collection_metadata={"hnsw:space": "cosine"},
    persist_directory=CHROMA_PATH
)

# Function to join the contexts
join_contexts = lambda contexts: "\n\n---\n\n".join([context.page_content for context in contexts])
show_sources = lambda contexts: [context.metadata.get("sources", None) for context in contexts]

# API 
app = FastAPI()

# When entering local host
@app.get("/")
async def root():
    return {"message": "Budi AI is running"}

# Allow the user to prompt for a response
@app.post("/chat/generate")
async def create_item(user_input: ChatPrompt):

    # Convert the input into dictionary
    user_input_dict = user_input.model_dump()

    # retrieve relevant context from vector database
    contexts = db.similarity_search(
        query=user_input_dict["question"],
        k=3
    )

    # Join the contexts and show sources
    context_text = join_contexts(contexts)
    sources = show_sources(contexts)

    # Format the input into a prompt
    prompt = prompt_template.format(context=context_text, question=user_input_dict["question"])

    # Generate response from model
    response = gen_model.invoke(prompt)
    response_text = f"{response}\nSources: {sources}"

    return {"id": user_input_dict["id"], "response": response_text}