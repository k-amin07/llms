from langchain_community.llms import Ollama

llm = Ollama(model="llama3")

## Simple prompt
resp = llm.invoke("How can langsmith help with testing?")
print(resp)

## Templated prompt
### Create a prompt template 
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class technical documentation writer"),
    ("user","{input}") # here input is a named parameter to the prompt.
])

### Create a simple LLM chain by combining the llm with prompt 
chain = prompt | llm
resp = chain.invoke({"input": "How can langsmith help with testing"})


## Output Parser
### Convert the output from "message" to a string using StrOutputParser
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

### Add to the chain 
chain = prompt | llm | output_parser 
resp = chain.invoke({"input": "How can langsmith help with testing"})
print(resp)

