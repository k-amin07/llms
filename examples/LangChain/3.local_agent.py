import os

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.tools import create_retriever_tool 
from langchain_core.messages import HumanMessage, AIMessage


if(os.getenv('OPENAI_API_KEY') == "ollama"):
    base_url="http://localhost:11434/v1"
    model = "llama3"
else:
    base_url = None
    model = "gpt-4o-mini"

embeddings = OllamaEmbeddings()

loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

retriever = vector.as_retriever()
retriever_tool = create_retriever_tool(retriever, name="langchain_docs_retriever", description="Retrieves docs from langchain hub")
tools = [retriever_tool]

llm = ChatOpenAI(model=model, base_url=base_url, verbose=True)
assistant_system_message = """You are a helpful assistant. \
Use tools (only if necessary) to best answer the users questions."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", assistant_system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

chat_history = []
user_message = input("Your Message: ")
while(user_message != "quit"):
    response = agent_executor.invoke({"input": user_message, "chat_history":chat_history})
    chat_history.append(HumanMessage(user_message))
    chat_history.append(AIMessage(response["output"]))
    user_message = input("Your Message: ")

