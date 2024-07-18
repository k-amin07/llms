from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.base import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import ToolException


def _handle_error(error: ToolException) -> str:
    return (
        "The following errors occurred during tool execution:"
        + error.args[0]
        + "Please try another tool."
    )

def multiply(a: int, b: int):
    return a * b

mul = StructuredTool.from_function(
    func=multiply,
    name="Multiply",
    description="useful for when you need to answer questions about multiplication",
    # coroutine= ... <- we can use this to specify async function as well.
    handle_tool_error=_handle_error
)

tools = [mul]

llm = ChatOpenAI(model="llama3", api_key="ollama", base_url="http://localhost:11434/v1", verbose=True)
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

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

chat_history = []
user_message = input("Your Message: ")
while(user_message != "quit"):
    response = agent_executor.invoke({"input": user_message, "chat_history":chat_history})
    chat_history.append(HumanMessage(user_message))
    chat_history.append(AIMessage(response["output"]))
    user_message = input("Your Message: ")
