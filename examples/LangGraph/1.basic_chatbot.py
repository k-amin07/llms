import os
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from typing import Annotated
from typing_extensions import TypedDict

# from IPython.display import Image, display

if(os.getenv('OPENAI_API_KEY') == "ollama"):
    base_url="http://localhost:11434/v1"
    model = "llama3"
else:
    base_url = None
    model = "gpt-4o-mini"

class State(TypedDict):
    # Here, messages have the type "list" and the add message function ensures 
    # that the new messages are added to the list, rather than replacing the 
    # old ones.
    messages: Annotated[list,add_messages]

graph_builder = StateGraph(State)

# At this point, our graph knows two things
# 1. Every node that we define will receive the current state as input and return a value that updates the state
# 2. messages will be appended to the current list, instead of overwriting the existing ones,
#    because of the add_messages functionn

# Now we will add initialize the LLM and create a chatbot
llm = ChatOpenAI(model=model, base_url=base_url, verbose=True)

# Create a basic function for the chatbot node. The function takes in the state and returns an updated messages list.
# the add messages function in the State class will append the llm response to the messages list.
def chatbot(state:State):
    return { "messages": [llm.invoke(state["messages"])] }

# add the chatbot node to the graph. The first argument is a unique name for the node
# the second argument is the function that gets called each time we use the node.
graph_builder.add_node("chatbot",chatbot)

# Next, we define the entry and exit points for our graph
# in this example, we only have one node so far, so we will add edges between it and the START and END node
# The add_edge function basically adds a directed edge, so the ordering of the inputs is important.
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot",END)

# Compile the graphh
graph = graph_builder.compile()

# Visualize the graph - we need IPython for it, so install it using pip
# Probably works better in jupyter notebooks, here it just outputs <IPython.core.display.Image object> in the terminal
# Look into this later, we can probably use Image from PIL
# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except:
#     pass

while(True):
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye")
        break
    print("")
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            print("Assistant: ", value["messages"][-1].content)
    print("\n**************\n")