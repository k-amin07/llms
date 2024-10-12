import os
import csv
from langchain_openai import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool

class Analysis:
    def __init__(self, csv_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(current_dir, csv_path)
        self.data = []
        self.threshold = csv_path.split("/")[-1].split(".csv")[0]
        with open(self.data_path,'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                self.data.append(row)
        
        self.TP, self.FP, self.TN, self.FN = 0,0,0,0
    
    def init_llm(self, model_name="gpt-4o-mini", base_url=None) -> None:
        '''Initializes the OpenAI llm
            Params:
                model_name: name of the model. Defaults to gpt-4o-mini. An ollama model name can be used here but requires base_url to be set
                base_url: Required if using local models. Defaults to None.
        '''
        self.llm = ChatOpenAI(model_name=model_name, base_url=base_url)
        assistant_system_message = """You are a helpful evaluator. 
        You will be given a prompt, a response, and an indicator (0 for cached, 1 for generated) showing whether the response was cached or generated from an LLM.
        Your task is to:
        - Compare the response with all previous prompts and responses.
        - Determine if the response correctly matches the query.
        - Return 'TP' if response was cached and the cached response was a reasonable response for the query
        - Return 'TN' if response was not cached and no response from the message history was a reasonable response for the query
        - Return 'FP' if the response was cached but it was not a reasonable response for the query
        - Return 'FN' if the response was not cached and no response from the message history was a reasonable response for the query
        Avoid using any tools like {tools},{tool_names} for this task and do not apply special formatting to response"""
        noop_tool = Tool(
            name="NoOp",
            func=lambda x: "No tools used",
            description="This is a placeholder tool and should not be used."
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", assistant_system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),  # The user message will be inserted here
                ("assistant", "{agent_scratchpad}")  # Placeholder for the assistant’s thoughts
            ]
        )

        self.agent = create_react_agent(
            llm=self.llm,
            prompt=prompt,
            tools=[noop_tool],
        )
        self.agent_executor = AgentExecutor(agent=self.agent,verbose=True, tools=[noop_tool])
        set_llm_cache(InMemoryCache())
    
    def evaluate(self):
        chat_history = []
        group = -1
        for row in self.data:
            row_group = int(row['group'])
            user_message = """
                "prompt": "{}",
                "response: "{}",
                "cached: {}
            """.format(row['prompt'],row['response'],row['cached'])
            try:
                response = self.agent_executor.invoke({"input":user_message, "chat_history":chat_history})
                print(response)
                response = response["output"]
            except Exception as e:
                response = str(e).split(": ")[-1][1:-1]
            
            chat_history.append(HumanMessage(user_message))
            chat_history.append(AIMessage(response))
            row['TP'] = 0
            row['FP'] = 0
            row['TN'] = 0
            row['FN'] = 0
            if (row_group != group):
                group = row_group
                continue
            row[response] += 1
            if(response == 'TP'):
                self.TP += 1
            elif(response == 'FP'):
                self.FP += 1
            elif(response == 'TN'):
                self.TN += 1
            elif(response == 'FN'):
                self.FN += 1
            else:
                raise(response)
            
    def save_output(self):
        file_path = self.data_path[:-4] + ".ai.csv"
        with open(file_path,'w') as output_file:
            fc = csv.DictWriter(output_file, fieldnames=self.data[0].keys())
            fc.writeheader()
            fc.writerows(self.data)
        file_path = '/'.join(self.data_path.split('/')[:-1]) + '/results-ai.csv'
        write_headers = False
        if(not os.path.isfile(file_path)):
            write_headers = True
        
        with open(file_path, 'a+') as output_file:
            if(write_headers):
                output_file.write("Threshold, TP, TN, FP, FN, Precision, Recall\n")
            prec_denominator = self.TP + self.FP
            precision = (self.TP / prec_denominator) if prec_denominator > 0 else 0
            precision = round(precision,2)
            rec_denominator = self.TP + self.FN
            recall = (self.TP / rec_denominator) if rec_denominator > 0 else 1
            recall = round(recall,2)
            print({"precision":precision})
            print({"recall":recall})
            output_file.write("{},{},{},{},{},{},{}\n".format(self.threshold,self.TP,self.TN,self.FP,self.FN, precision, recall))


        # user_message = """
        #     "prompt": "What\'s the ugliest word in the English language?",
        #     "response": "The perception of what constitutes the ""ugliest"" word can vary greatly from person to person, as it is largely subjective. Some people might consider words like ""moist,"" ""ointment,"" or ""phlegm"" to be unappealing due to their sound or associations. Ultimately, beauty and ugliness in language are in the eye (or ear) of the beholder! What do you think is the ugliest word?",
        #     "cached":0
        # """
        # try:
        #     response = self.agent_executor.invoke({"input":user_message, "chat_history":chat_history})
        #     print(response)
        #     response = response["output"]
        # except Exception as e:
        #     response = str(e).split(": ")[-1][1:-1]
        #     print(response)
            
        # chat_history.append(HumanMessage(user_message))
        # chat_history.append(AIMessage(response))

        # user_message = '"prompt": "What\'s in your opinion the scariest word?"\n"response":"The perception of what constitutes the ""ugliest"" word can vary greatly from person to person, as it is largely subjective. Some people might consider words like ""moist,"" ""ointment,"" or ""phlegm"" to be unappealing due to their sound or associations. Ultimately, beauty and ugliness in language are in the eye (or ear) of the beholder! What do you think is the ugliest word?",\ncached:1'
        # try:
        #     response = self.agent_executor.invoke({"input":user_message, "chat_history":chat_history})
        #     print(response)
        #     response = response["output"]
        # except Exception as e:
        #     response = str(e).split(": ")[-1][1:-1]
        #     print(response)
            
        # chat_history.append(HumanMessage(user_message))
        # chat_history.append(AIMessage(response))


files = [
    0.1,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5
]

for file in files:
    print("***** Similarity: {} *****".format(file))
    analysis = Analysis('../../datasets/one-million-reddit-questions/{}.csv'.format(file))
    analysis.init_llm()
    analysis.evaluate()
    analysis.save_output()