import os
import json

from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage, AIMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START
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


tavily_tool = TavilySearchResults(max_results=2)
tools = [tavily_tool]

llm = ChatOpenAI(model=model, base_url=base_url)
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
graph_builder.add_edge(START, "chatbot")
memory = SqliteSaver.from_conn_string(":memory:")
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"]
)

user_input = "I'm learning LangGraph. Could you do some research on it for me?"
config = {"configurable": {"thread_id": "1"}}

events = graph.stream({"messages": [("user", user_input)]}, config)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

snapshot = graph.get_state(config)
existing_message = snapshot.values["messages"][-1]
print("Original")
print("Message ID", existing_message.id)
print(existing_message.tool_calls[0])

new_tools_call = existing_message.tool_calls[0].copy()
new_tools_call["args"]["query"] = "LangGraph Human-in-the-loop workflow"
new_message = AIMessage(
    content=existing_message.content,
    tool_calls=[new_tools_call],
    id=existing_message.id
)
## The id tells langgraph which message is to be replaced

print("Updated")
print("Message ID", new_message.id)
print(new_message.tool_calls[0])
graph.update_state(config, {"messages": [new_message]})

print("\n\nNew Tools call")
print(graph.get_state(config).values["messages"][-1].tool_calls)

events = graph.stream(None,config,stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

## Check if the graph remembers our updated query

events = graph.stream(
    {
        "messages": ("user", "Remember what I'm learning about?",)
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()


