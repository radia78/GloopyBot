# Import libraries for the discord bot API
import os
from discord import Intents
from discord.ext import commands
from dotenv import load_dotenv

# Improt libraries related to the LLM feature
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings
from langchain_community.retrievers import ArxivRetriever
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from utils import *

# Load the environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN") # Obtain the Discord bot token
GUILD = os.getenv("DISCORD_GUILD") # Obtain the Discord bot guild
EMB_MODEL_PATH = os.getenv("EMB_MODEL_PATH")
GEN_MODEL_PATH = os.getenv("GEN_MODEL_PATH")

# Initialize Arxiv retriever to get papers & text splitter
""" arxiv_retriever = ArxivRetriever(load_max_docs=1)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False
)
 """
# Initialize the LLM MODELS
callback_manager = CallbackManager([StreamingStdOutCallbackHandler])

# Initialize the generator
gen_model = LlamaCpp(
    model_path=GEN_MODEL_PATH,
    max_tokens=2048,
    n_gpu_layers=-1,
    n_ctx=512,
    callback_manager=callback_manager,
    verbose=True
)

# Initialize the embedding model for the vector database
#emb_model = LlamaCppEmbeddings(model_path=EMB_MODEL_PATH)

# Function to join the contexts
#join_contexts = lambda contexts: "\n\n---\n\n".join([context.page_content for context in contexts])
#show_sources = lambda contexts: [context.metadata.get("sources", None) for context in contexts]

intents = Intents(
    messages=True,
    guilds=True,
    message_content=True
)

bot = commands.Bot(
    intents=intents,
    command_prefix='!'
)

# Event when first loaded up
@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected to Discord!')

# Summarize the document attached to the message
@bot.command(name='prompt')
async def summarize(ctx, *, message: str):
    prompt = prompt_template.format(question=message)
    response = gen_model.invoke(prompt)
    await ctx.send(response)

# The QA command
@bot.command(name='qa')
async def question_answer(ctx):
    pass

# Self-destruct command
@bot.command(name='self_destruct')
async def poke(ctx):
    await ctx.send("NO PLEASE! I DON'T WANNA DIE!")

bot.run(TOKEN)