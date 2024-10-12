import csv
import os

class ResponseAnalyzer:
    '''
    Manually analyze queries and mark whether it should be a cache hit/miss
    '''
    def __init__(self, csv_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(current_dir, csv_path)
        self.data = []
        self.threshold = csv_path.split("/")[-1].split(".csv")[0]
        with open(self.data_path,'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                self.data.append(row)

    def print(self):
        print(self.data)
    
    def user_input(self):
        queries = []
        for row in self.data:
            query = row['prompt']
            del row['response']
            del row['cached']
            del row['group']
            queries.append(query)
            print("Prompt: ", query)
            user_input = int(input("Enter 0 if this should not be cached, 1 if it should be\n"))
            row['should_cache']=user_input
            print("********************")
            print("Group Queries: ", queries)
        
    def save_output(self):
        file_path = self.data_path[:-7] + "user_analysis.csv"
        with open(file_path,'w') as output_file:
            fc = csv.DictWriter(output_file, fieldnames=self.data[1].keys())
            fc.writeheader()
            fc.writerows(self.data)





ra = ResponseAnalyzer('../../datasets/one-million-reddit-questions/0.1.csv')
ra.user_input()
ra.save_output()