########ChatModel
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-5.5", temperature=0)

response = model.invoke("Explain photosynthesis in one sentence.")
print(response.content)

#####################################################################
###Prompt Templates
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise tutor. Answer in exactly one sentence."),
    ("user", "Explain {topic} to a {audience}."),
])

messages = prompt.invoke({"topic": "photosynthesis", "audience": "five-year-old"})
print(messages)  # a structured list of messages, ready to send to a model

####################################################################################
##########Structured output
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

class MovieReview(BaseModel):
    title: str = Field(description="The movie's title")
    rating: int = Field(description="Score from 1 to 10")
    summary: str = Field(description="One-sentence summary")

model = init_chat_model("openai:gpt-5.5")
structured_model = model.with_structured_output(MovieReview)

result = structured_model.invoke("Review the movie Inception.")
print(result.title, result.rating)   # typed Python object, not a string
print(type(result))                  # <class '__main__.MovieReview'>

######################################################################################
####LCEL
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise tutor."),
    ("user", "Explain {topic} in one sentence."),
])
model = init_chat_model("openai:gpt-5.5")
parser = StrOutputParser()   # pulls the plain string out of the model's response

chain = prompt | model | parser

print(chain.invoke({"topic": "photosynthesis"}))
####################################################################################
#############RAG##################################################################
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

# 1. Load
docs = TextLoader("my_notes.txt").load()

# 2. Split
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)

# 3. Embed + 4. Store (InMemory is fine for learning; use a real DB in production)
vectorstore = InMemoryVectorStore.from_documents(chunks, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Build the RAG chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the context below. If unsure, say you don't know.\n\nContext:\n{context}"),
    ("user", "{question}"),
])
model = init_chat_model("openai:gpt-5.5")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# Wire it together: retrieve, format, fill prompt, call model, parse
rag_chain = (
    {"context": retriever | format_docs, "question": lambda x: x}
    | prompt
    | model
    | StrOutputParser()
)

print(rag_chain.invoke("What did my notes say about the budget meeting?"))
################################################################################
#Tools
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return f"It's 22°C and sunny in {city}."

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

agent = create_agent(
    model="openai:gpt-5.5",
    tools=[get_weather, multiply],
    system_prompt="You are a helpful assistant. Use tools when they help.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "What's the weather in Tokyo, and what's 23 times 7?"}]})
print(result["messages"][-1].content)

######################################################3333
#Check Pointer
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="openai:gpt-5.5",
    tools=[],
    checkpointer=InMemorySaver(),   # remembers state across turns
)

config = {"configurable": {"thread_id": "user-123"}}
agent.invoke({"messages": [{"role": "user", "content": "My name is Sam."}]}, config)
reply = agent.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, config)
print(reply["messages"][-1].content)  # knows it's "Sam"

########################################################################
#Logging
call_count = {"count": 0}

def logging_middleware(func):
    """Wraps a tool to log its calls."""
    def wrapper(*args, **kwargs):
        call_count["count"] += 1
        print(f"  [Tool call #{call_count['count']}] {func.__name__}({args}, {kwargs})")
        result = func(*args, **kwargs)
        print(f"  [Result] {result}")
        return result
    return wrapper

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b

# Create a simple chain that uses tools with logging
math_agent = create_agent(
    model=model,
    tools=[add, subtract],
    system_prompt="You are a math helper. Use tools for calculations.",
)

# Wrap the agent's invoke to log inputs/outputs
original_invoke = math_agent.invoke

def invoke_with_logging(input_data, config=None):
    print(f"[Input] {input_data.get('messages', [{}])[-1].get('content', '')}")
    result = original_invoke(input_data, config)
    print(f"[Output] {result['messages'][-1].content}")
    return result

math_agent.invoke = invoke_with_logging

result = math_agent.invoke({"messages": [{"role": "user", "content": "What's 100 minus 37?"}]})
print(f"\nFinal answer: {result['messages'][-1].content}") 




