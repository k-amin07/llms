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
Question: {input}""")
document_chain = create_stuff_documents_chain(llm, prompt)

###################################################
## Run directly with the two named parameters
from langchain_core.documents import Document
document_chain.invoke({
    "input": "how can langsmith help with testing?",
    "context": [Document(page_content="langsmith can let you visualize test results")]
})

## OR better way: use retrieval chain
from langchain.chains import create_retrieval_chain
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"}) # returns a dict
print(response["answer"])
