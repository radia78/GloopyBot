from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    
    {context}
    
    ---
    
    Answer the question based only on the above context: {question}
    
    """

# Data structure for chat prompts
class ChatPrompt(BaseModel):
    id: int
    question: str

# Initialize the prompt template
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)