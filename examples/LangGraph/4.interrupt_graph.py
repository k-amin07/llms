import os
import json

from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
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

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""
    def __init__(self, tools: list) -> None:
        # store each tool against its name in a dict
        self.tools_by_name = {tool.name: tool for tool in tools}
    
    def __call__(self, inputs: dict):
        # python 3.8+ allows variable assignments within expressions like if, while etc
        if messages := inputs.get("messages",[]):
            # for an empty list, this would normally raise IndexError, 
            # but this syntax allows us to raise a custom Value error
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        
        print(message)
        outputs = []
        for tool_call in message.tool_calls:
            # get the tool from tools_by_name dict and invoke it with the relevant args
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                )
            )
        
        return {"messages": outputs} 



graph_builder = StateGraph(State)

llm = ChatOpenAI(model=model, base_url=base_url)

tavily_tool = TavilySearchResults(max_results=2)
tools = [tavily_tool]

llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

# tool_node = ToolNode(tools=[tavily_tool])
tool_node = BasicToolNode(tools=[tavily_tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge("tools", "chatbot")
# graph_builder.add_edge(START, "chatbot")
graph_builder.set_entry_point("chatbot")

memory = SqliteSaver.from_conn_string(":memory:")
### Everything else remains the same, but now wer can interrupt the llm execution 
# at a specified point
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"],
)

## we need a thread id for the llm to identify this conversation
config = {"configurable": {"thread_id": "1"}}

user_input = "I'm learning LangGraph. Could you do some research on it for me?"
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

snapshot = graph.get_state(config)
print(snapshot.next) # should be (action,)
## well its actually tools in my case which makes more sense

existing_message = snapshot.values["messages"][-1]
print(existing_message.tool_calls)

## We can let the graph continue its thing
## None will not append anything new to the current state

events = graph.stream(None, config=config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()