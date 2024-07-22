import os
import json

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from typing import Annotated, Literal
from typing_extensions import TypedDict

if(os.getenv('OPENAI_API_KEY') == "ollama"):
    base_url="http://localhost:11434/v1"
    model = "llama3"
else:
    base_url = None
    model = "gpt-4o-mini"

class State(TypedDict):
    messages: Annotated[list,add_messages]

## We will later replace this with langgraph's prebuilt tool node
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

llm = ChatOpenAI(model=model, base_url=base_url, verbose=True)

tavily_tool = TavilySearchResults(max_results=2)
tools = [tavily_tool]

llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot",chatbot)

## Add the BasicToolNode to the graph
## We are also binding the tool with LLM, but we have to add it to the graph as well
tool_node = BasicToolNode(tools=[tavily_tool])
graph_builder.add_node("tools", tool_node)

## We will now add a conditional edge which 
# will route to tools if tool calls are present and "__end__" if not.

def route_tools(state: State) -> Literal["tools", "__end__"]:
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        # If the passed state is list type, get the last message
        aimessage = state[-1]
    elif messages := state.get("messages",[]):
        # Otherwise if its dict type, get messages and return the last message
        aimessage = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    ## If the AI message called the tool, route to tools, otherwise route to end
    if(hasattr(aimessage,"tool_calls") and len(aimessage.tool_calls) > 0):
        return "tools"
    return "__end__"

graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # the dictionary below has the mapping of the form
    # key: output of the condition (i.e. output of route_tools)
    # value: name of the node. by default it is the same as the key, 
    # but can be changed to something else. e.g "tools": "my_tools"
    {"tools":"tools", "__end__":"__end__"}
    # we are basically adding a conditional edge, either to rools or to end,
    # depending on the output of route_tools
)

graph_builder.add_edge("tools", "chatbot")
# ^ every time tools are called, return to chatbot
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

graph.get_graph().print_ascii()

user_input = "I'm learning LangGraph. Could you do some research on it for me?"
events = graph.stream(
    {"messages": [("user", user_input)]}, stream_mode="values"
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# from langchain_core.messages import BaseMessage
# while(True):
#     user_input = input("User: ")
#     if user_input.lower() in ["quit", "exit", "q"]:
#         print("Goodbye")
#         break
#     print("")
#     for event in graph.stream({"messages": [("user", user_input)]}):
#         for value in event.values():
#             if isinstance(value["messages"][-1], BaseMessage):
#                 print("Assistant:", value["messages"][-1].content)
# Now if a question is outside of the assistant's training data, it will use the tools