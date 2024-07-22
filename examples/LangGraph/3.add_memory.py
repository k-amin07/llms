## In the previous example, if we try to reason about a response, the chatbot has no memory of it.
## So we are going to add that

## This is also known as checkpointing, and it is much more powerful than simple chat memory
## It can allow error recovery, for instance

from langgraph.checkpoint.sqlite import SqliteSaver
memory = SqliteSaver.from_conn_string(":memory:")
# This saves everything in memory, can be replaced by an actual database

## In this part, we are going to use the functions from langgraph, rather than implementing them on our own


import os
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

if(os.getenv('OPENAI_API_KEY') == "ollama"):
    base_url="http://localhost:11434/v1"
    model = "llama3"
else:
    base_url = None
    model = "gpt-4o-mini"


class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatOpenAI(model=model, base_url=base_url, verbose=True)
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

## The above code is essentially the same as enhance with tools part
## However, we are going to compile the graph with checkpointer, 
# which is the db/memory that we initialized earlier
graph = graph_builder.compile(checkpointer=memory)

## we need a thread id for the llm to identify this conversation
config = {"configurable": {"thread_id": "1"}}

user_input = "Hi there! My name is Will."
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    event["messages"][-1].pretty_print()

user_input = "Remember my name?"

## The llm should remember the name at this point
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    event["messages"][-1].pretty_print()

## If we use a different thread_id, it wont remember the name
events = graph.stream(
    {"messages": [("user", user_input)]},
    {"configurable": {"thread_id": "2"}},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()

## So essentially, by only changing the thread_id, we can have 
# an entirely new instance of the chat that has no memory of the 
# previous one

## We can also fetch the current state of the graph for a given config
snapshot = graph.get_state(config)
print(snapshot)
print(snapshot.next)

# If the graph is not in the end state, it will show the next state
# in this case, snapshot.next would be empty
