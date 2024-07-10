from langchain_community.llms import Ollama
llm = Ollama(model="llama2")

# Providing addtional context
## We can provde additional context to the LLM toWe chain the prompt with the llm
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()

## Create Embeddings using OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings()

## Ingest the embeddings into the FAIS vector store
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)


## Create Document Chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    Question: {input}
""")
document_chain = create_stuff_documents_chain(llm, prompt)

###############################################################
## Run directly with the two named parameters
print("#" * 50)
print("## Run directly with the two named parameters\n")

from langchain_core.documents import Document
response = document_chain.invoke({
    "input": "how can langsmith help with testing?",
    "context": [Document(page_content="langsmith can let you visualize test results")]
})

print(response)

###############################################################
## OR better way: use retrieval chain
print("\n" + "#" * 50)
print("## OR better way: use retrieval chain\n")

from langchain.chains import create_retrieval_chain
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"}) # returns a dict

print(response["answer"])

###############################################################
## Create a conversational retrieval chain
print("\n" + "#" * 50)
print("## Create a conversational retrieval chain\n")
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user","{input}"),
    ("user","Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])

retriever_chain = create_history_aware_retriever(llm,retriever,prompt)

## Test the conversational retrieval chain
from langchain_core.messages import HumanMessage, AIMessage
chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
response = retriever_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})

print(response)

###############################################################
## Create a new chain with retrieved documents
print("\n" + "#" * 50)
print("## Create a new chain with retrieved documents\n")
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain) # can also use the retriever chain here

chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
response = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})

print(response["answer"])