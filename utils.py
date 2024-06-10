from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
    You are an AI assistant. Answer the question based on your own knowledge.
    ---
    question: {question}
    
    """

# Data structure for chat prompts
class ChatPrompt(BaseModel):
    id: int
    paper_id: str
    question: str

# Initialize the prompt template
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)