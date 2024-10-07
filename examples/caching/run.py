import csv
import json
import os
import pickle

from redisvl.extensions.llmcache import SemanticCache
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Caching:
    '''Caching for LLMs
    Typical workflow:
    worker = Caching(dataset_path='/path/to/dataset_or_question_group',redis_url='redis://host:port',similarity_threshold=0.7)
    worker.init_cache(name="llmcache",prefix="llmcache",distance_threshold=0.25, use_sim_threshold=False)
    worker.init_llm(model_name="gpt-4o-mini", base_url=None)
    worker.load_data(question_key='question')
    worker.save_question_groups() # or worker.save_question_groups_of_at_least_2()
    worker.process

    '''
    def __init__(self, dataset_path, redis_url="redis://localhost:6379", similarity_threshold=0.7) -> None:
        '''Initializes the caching class
            Params: 
                dataset_path (required): path to the dataset used.
                redis_url (optional): redis connection string - defaults to redis://localhost:6379
                similarity_threshold: cosine distance threshold to compute similarity between sentences.
                question_key: the key against which questions are stored in the dataset. Defaults to question
        '''
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(current_dir, dataset_path)
        self.redis_url = redis_url
        self.results = []
        self.questions = []
        self.groups = []
        self.similarity_threshold = similarity_threshold
        self.total_queries = 0
        self.cached_queries = 0


    def init_cache(self, name="llmcache",prefix="llmcache",distance_threshold=0.25, use_sim_threshold=False)->None:
        '''Initializes the Redis Semantic Cache
            Params:
                name: Name for the redis collection. Defaults to llmcache
                prefix: prefix for keys in llm cache. Defaults to 
                distance_threshold: custom threshold used if use_sim_threshold is set to False.
                use_sim_threshold: Whether to use the same similarity threshold as the caching class
        '''
        
        self.distance_threshold = distance_threshold

        self.llmcache = SemanticCache(
            name=name,
            prefix=prefix,
            redis_url=self.redis_url,
            distance_threshold=1-self.similarity_threshold if use_sim_threshold else distance_threshold
        )
    
    def init_llm(self, model_name="gpt-4o-mini", base_url=None) -> None:
        '''Initializes the OpenAI llm
            Params:
                model_name: name of the model. Defaults to gpt-4o-mini. An ollama model name can be used here but requires base_url to be set
                base_url: Required if using local models. Defaults to None.
        '''
        self.llm = ChatOpenAI(model_name=model_name, base_url=base_url)

    def extract_questions(self, key='questions'):
        '''Extracts the questions from the dataset
        '''
        for row in self.data:
            self.questions.append(row[key])
    
    def save_question_groups(self):
        '''
        Function to save the quesstion groups in the same path as the dataset
        '''
        output_path = '/'.join(self.data_path.split('/')[:-1]) + '/question_groups.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(self.groups, f)
    
    def save_question_groups_of_at_least_2(self):
        '''
        Function to save the quesstion groups that have at least 2 or more elements in the same path as the dataset
        '''
        output_path = '/'.join(self.data_path.split('/')[:-1]) + '/question_groups_2_or_more.pkl'
        
        result_groups = []
        for group in self.groups:
            if(len(group) > 1):
                result_groups.append(group)
        with open(output_path, 'wb') as f:
            pickle.dump(result_groups, f)

    def load_data(self, question_key='question'):
        '''
        Loads the dataset from self.data_path path. If the filepath ends with jsonl, the function loads and processes the json data. If it ends with .pkl, it assumes the data to be processed and loads it.
        Params:
            question_key: the key against which questions are stored in the dataset. Defaults to question
        '''

        def load_jsonl(file_path):
            '''Loads the dataset (jsonl format) from a given path.'''
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data
        
        def load_csv(file_path):
            '''Loads dataset (csv format) from a given path'''
            data = []
            with open(file_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    data.append(row)
            return data


        def load_question_groups(filepath):
            '''
            Function to load question groups from the given path
            '''
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        
        file_path = self.data_path

        if(file_path.endswith('.pkl')):
            self.groups = load_question_groups(file_path)
            return

        if(file_path.endswith(".jsonl")):
            self.data = load_jsonl(file_path)
        elif(file_path.endswith('.csv')):
            self.data = load_csv(file_path)

        self.extract_questions(question_key)
        self.compute_question_similarity()
        
    
    def print_data(self):
        '''
        Prints the loaded dataset
        '''
        print(json.dumps(self.data, indent=4, ensure_ascii=False))

    def print_groups(self):
        '''
        Prints the groups of similar questions
        '''
        print(self.groups)
        print(len(self.groups))


    def compute_question_similarity(self):
        '''
        Groups similar questions based on cosine similarity.
        Generates the embeddings for the questions in the dataset and groups
        similar questions together. Similarity threshold is defined during the class initialization.
        '''
        model = SentenceTransformer('all-mpnet-base-v2')
        self.questions = self.questions[:10000]
        embeddings = model.encode(self.questions)
        self.cosine_sim_matrix = cosine_similarity(embeddings)
        used = set()
        for i in range(len(self.questions)):
            if i in used: 
                continue

            group = [self.questions[i]]
            used.add(i)
            for j in range(i+1, len(self.questions)):
                if j not in used and self.cosine_sim_matrix[i][j] > self.similarity_threshold:
                    group.append(self.questions[j])
                    used.add(j)
            self.groups.append(group)
    

    def process_data(self, limit:int=20, process_all:bool=False):
        '''
        Goes through the questions groups and checks whether its a cache hit. Sends to LLM if cache misses
        Params:
            limit: number of groups to process. defaults to 20
            process_all: ignores the limit if set to True.
        '''

        if process_all or limit > len(self.groups):
            limit = len(self.groups)

        for idx, group in enumerate(self.groups):
            if(idx == limit):
                break
            print("Processing group {}".format(idx))
            for prompt in group:
                if resp := self.llmcache.check(prompt=prompt):
                    response = resp[0]["response"]
                    self.append_results(prompt, response, idx, was_cached=True)
                else:
                    resp = self.llm.invoke(prompt)
                    self.llmcache.store(
                        prompt=prompt,
                        response=resp.content
                    )
                    response = resp.content
                    self.append_results(prompt,resp.content, idx, was_cached = False)
        self.save_results()
        self.percent_match(limit)
    
    def append_results(self, prompt:str, response:str, group:int, was_cached:bool=False):
        '''
        Appends a dict of prompt, response and whether the response was cached to the results array.
        May get changed later.
        '''
        self.results.append({"prompt": prompt, "response": response, "cached": int(was_cached), "group": group})
        if(was_cached):
            self.cached_queries += 1
        self.total_queries += 1

                
    def save_results(self):
        '''
        Saves results to csv file in the same directory as dataset
        '''
        file_path = '/'.join(self.data_path.split('/')[:-1]) + '/' + str(self.distance_threshold) + '.csv'
        keys = self.results[0].keys()

        with open(file_path, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.results)

    def percent_match(self, num_groups:int):
        '''
        Prints the percentage of queries that were cached
        '''
        max_cached_queries = self.total_queries-num_groups # the first query in every group will never be cached
        percent_cached = round(self.cached_queries/max_cached_queries * 100,2)
        print('Total Queries: {}\nTotal Groups: {}\nCached Queries: {}\nPercent Cached:{}'.format(self.total_queries, num_groups, self.cached_queries, percent_cached))
        file_path = '/'.join(self.data_path.split('/')[:-1]) + '/results.csv'
        write_headers = False
        if(not os.path.isfile(file_path)):
            write_headers = True

        with open(file_path, 'a+') as output_file:
            if(write_headers):
                output_file.write('Threshold, Total Groups, Total Queries, Max Cached Queries, Cached Queries, Percent Cached\n')
            output_file.write("{},{},{},{},{},{}\n"
                              .format(self.distance_threshold, num_groups, self.total_queries, max_cached_queries, self.cached_queries, percent_cached))

    def close(self):
        '''
        Clears the redis cache for next experiment
        '''
        self.llmcache.delete()


# # experiment = Caching(dataset_path='../../datasets/reddit-qa/train.jsonl')
# # experiment = Caching(dataset_path='../../datasets/one-million-reddit-questions/one-million-reddit-questions.csv')
# experiment = Caching(dataset_path='../../datasets/one-million-reddit-questions/question_groups.pkl')

# experiment.init_cache()
# # experiment.init_llm("llama3","http://localhost:11434/v1")
# experiment.init_llm()
# # experiment.print_data()
# experiment.load_data(question_key='title')
# experiment.save_question_groups()
# experiment.save_question_groups_of_at_least_2()
# # experiment.print_groups()
# experiment.process_data()

for i in [0.1,0.2,0.25,0.3,0.35,0.4,0.45,0.5]:
    experiment = Caching(dataset_path='../../datasets/one-million-reddit-questions/question_groups_2_or_more.pkl')
    experiment.init_cache(prefix=str(i),distance_threshold=i)
    experiment.init_llm()
    experiment.load_data(question_key='title')
    experiment.process_data()
    experiment.close()


##############
# TO DO
# Step 1: Use BERT or cosine similarity to find similar questions.
# Step 2: Run the questions throught the code to see if there is a cache hit or miss
# Step 3: Repeat with varying distance threshold

# Repeat for this dataset: https://huggingface.co/datasets/SocialGrep/one-million-reddit-questions?row=28
# I saw some similar questions in a quick glance.
###############