import os
import csv

class UserAnalysis:
    '''
    Compares cache results with the user input on whether the query should have been cached
    '''
    def __init__(self, csv_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.ua_path = os.path.join(current_dir,csv_path.split('0')[0] + 'user_analysis.csv')
        self.data_path = os.path.join(current_dir, csv_path)
        self.data = []
        self.user_analysis = []
        self.threshold = csv_path.split("/")[-1].split(".csv")[0]
        with open(self.data_path,'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                self.data.append(row)
        with open(self.ua_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                self.user_analysis.append(row)
        self.TP, self.FP, self.TN, self.FN = 0,0,0,0
    
    def evaluate(self):
        group = -1
        for index,row in enumerate(self.data):
            row_group = int(row['group'])
            row['TP'] = 0
            row['FP'] = 0
            row['TN'] = 0
            row['FN'] = 0
            if (row_group != group):
                group = row_group
                continue
            user_result = self.user_analysis[index]
            user_response = int(user_result['should_cache'])

            was_cached = int(row['cached'])
            if user_response == 1 and was_cached == 1:
                self.TP += 1
                row['TP'] += 1
            elif user_response == 0 and was_cached == 0:
                self.TN += 1
                row['TN'] = 1
            elif user_response == 0 and was_cached == 1:
                self.FP += 1
                row['FP'] = 1
            elif user_response == 1 and was_cached == 0:
                self.FN += 1
                row['FN'] = 1
    def save_output(self):
        file_path = self.data_path[:-4] + ".ui.csv"
        with open(file_path,'w') as output_file:
            fc = csv.DictWriter(output_file, fieldnames=self.data[1].keys())
            fc.writeheader()
            fc.writerows(self.data)
        file_path = '/'.join(self.data_path.split('/')[:-1]) + '/results-ui.csv'
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
            output_file.write("{},{},{},{},{},{},{}\n".format(self.threshold,self.TP,self.TN,self.FP,self.FN, precision, recall))


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
    ra = UserAnalysis('../../datasets/one-million-reddit-questions/{}.csv'.format(file))
    ra.evaluate()
    ra.save_output()