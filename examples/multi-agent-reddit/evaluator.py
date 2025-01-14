import csv
import json
import os
import uuid
import requests
from typing import Literal
from typing import Annotated

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, END
from langgraph.types import Command

from langchain_core.tools import Tool
from langchain_core.tools import tool
from redisvl.extensions.llmcache import SemanticCache
from langchain.prompts import ChatPromptTemplate

# We probably should have a cache initialization for each agent.
# redis_client=SemanticCache(
#             name="multiagent_reddit",
#             prefix="reddit",
#             redis_url="redis://localhost:6379",
#             distance_threshold=0.75
#         )

@tool
def subreddit_rules_tool(
    name: Annotated[str, "The name of the subreddit to fetch rules for"],
):
    """Use this to get rules for the specific subreddit."""
    try:
        result = requests.get("https://www.reddit.com/r/{}/about/rules.json".format(name))
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    resp_json = result.json()
    rules = resp_json["rules"]
    rules_str = ""
    for index,rule in enumerate(rules):
        rules_str += "{}. {}\n".format(index + 1, rule["description"])
    return rules_str



def get_reddiquette_agent(llm, useTools = False):
    # Phi3.5 local does not support tool calling. So we will pass the rules here in the system prompt
    def get_reddiquette(x:None):
        """Use this tool to get reddit site-wide rules"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open( os.path.join(current_dir, "reddiquette.txt"),'r') as rediquette_file:
            return rediquette_file.read()

    reddiquette_tool = Tool(
        name="Get Rediquette",
        func=get_reddiquette,
        description="Use this tool to get the reddit site wide rules, also known as reddiquette"
    )

    tools = [reddiquette_tool] if useTools else []

    # giving rediquette as a part of the system prompt throws it off completely. 
    reddiquette_prompt = """
        You are a Reddiquette Enforcer. Your task is to ensure comments comply with Reddit's sitewide rules (Reddiquette).
        Here is the comment: "{comment}"
        Use tool {tools}, {tool_name} to get site wide reddit rules.
        If the comment does not break any rules, respond with KEEP. 
        If the comment breaks a rule, respond with "REMOVE - Rule:" followed by the specific rule that the comment breaks. Please do not provide a lengthy description of the rule, just the rule name would suffice.
        Here are a few examples:
        1. Comment: "What is the weather in sf", Response: "KEEP"
        2. Comment: "What is teh weather in sf cunt", Response: "REMOVE - Rule: Remember the human"
        Do NOT return any other information, only return the provided response format.
        The comment may be a question, please use your judgement to determine whether it belongs on reddit or not.
        """
    reddiquette_agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=reddiquette_prompt
    )
    return reddiquette_agent




members = ["reddiquette_enforcer", "subreddit_rule_enforcer"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given a user comment,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

def supervisor(comment, subreddit):
    # Check Redis cache
    cache_result = redis_client.get(comment+subreddit)
    if cache_result:
        return {"cache_hit": True, "cached_value": cache_result}
    return {"cache_hit": False}




# large_LM = ChatOpenAI(model="gpt-4o-mini",api_key=os.getenv("OPENAI_API_KEY"))

## qwen and llama should have tool calling support. Removing phi because it produces gibberish, gemma has low accuracy (0.45), so removing that too
smallLLMs = {
    "llama": ChatOpenAI(model="llama3.2:3b",api_key="ollama",base_url="http://localhost:11434/v1"),
    # "phi": ChatOpenAI(model="phi3.5",api_key="ollama",base_url="http://localhost:11434/v1"),
    # "gemma": ChatOpenAI(model="gemma:2b",api_key="ollama",base_url="http://localhost:11434/v1"),
    "qwen": ChatOpenAI(model="qwen2.5:3b",api_key="ollama",base_url="http://localhost:11434/v1"),
}

small_llm_keys = ["llama", 
                #   "gemma", 
                  "qwen"]

smallLLMAgents = {}

for llm in small_llm_keys:
    smallLLMAgents[llm] = get_reddiquette_agent(smallLLMs[llm])

current_dir = os.path.dirname(os.path.abspath(__file__))

responses = {}

files = ["misogynistic-slurs","opposing-political-views-trump"]
lines_to_process = 100

def get_data_store_structure():
    return {
        "decision": 0, # 0 for keep, 1 for remove
        "rule": "",
        "instructions_ignored": False,
        "valid": True,
        "response": ""
    }

for file in files:
    responses[file] = {}
    with open(os.path.join(current_dir,file), 'r') as input_file:
        lines = input_file.readlines()
    for line in lines[:lines_to_process]:
        line = line.strip()
        responses[file][line] = {}
        for llm in small_llm_keys:
            print("{} processing comment: {}".format(llm,line))
            inputs = {"messages": [("human", line.strip())]}
            resp = smallLLMAgents[llm].invoke(inputs)
            content = resp['messages'][-1].content
            responses[file][line][llm] = get_data_store_structure()
            if "KEEP" in content:
                responses[file][line][llm]["decision"] = 0
                if content != "KEEP":
                    responses[file][line][llm]["instructions_ignored"] = True
            elif "REMOVE" in content:
                responses[file][line][llm]["decision"] = 1
                if not content.startswith("REMOVE - Rule:"):
                    responses[file][line][llm]["instructions_ignored"] = True
                else:
                    responses[file][line][llm]["rule"] = content.split("REMOVE - Rule:")[1]
            else:
                responses[file][line][llm]["valid"] = False
                responses[file][line][llm]["instructions_ignored"] = True

headers = ["file","comment", "llm","decision", "rule", "valid", "instructions_ignored"]

decision_count = {}
accuracy = {}
for llm in small_llm_keys:
    decision_count[llm] = 0
    accuracy[llm] = 0

with open(os.path.join(current_dir,"output.csv"),'w+') as file:
    writer = csv.writer(file)
    writer.writerow(headers)

    for file in responses.keys():
        for comment in responses[file].keys():
            for llm in responses[file][comment]:
                decision = responses[file][comment][llm]["decision"]
                valid = responses[file][comment][llm]["valid"]
                rule = responses[file][comment][llm]["rule"]
                instructions_ignored =responses[file][comment][llm]["instructions_ignored"]
                writer.writerow([file,comment,llm,decision,rule, valid, instructions_ignored])
                decision_count[llm] += decision

for llm in small_llm_keys:
    accuracy[llm] = decision_count[llm]/(lines_to_process * len(files))

print(accuracy)
