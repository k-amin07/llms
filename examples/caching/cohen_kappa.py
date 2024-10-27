import numpy as np
import pandas as pd
import os
import sklearn.metrics
import csv


current_dir = os.path.dirname(os.path.abspath(__file__))

user_path = '../../datasets/one-million-reddit-questions/results-ui.csv'
user_path = os.path.join(current_dir,user_path)
ai_path = '../../datasets/one-million-reddit-questions/results-ai.csv'
ai_path = os.path.join(current_dir,ai_path)

df_user = pd.read_csv(user_path)
df_ai = pd.read_csv(ai_path)
# print(df_ai['TP'])
# print(df_user['TP'])
df_user.columns = df_user.columns.str.strip()
df_ai.columns = df_ai.columns.str.strip()

kappa_scores = []

def expand_counts_to_labels(tp, tn, fp, fn):
    return [1] * int(tp) + [0] * int(tn) + [2] * int(fp) + [3] * int(fn)

def cohen_kappa_using_sklearn(tp1, tn1, fp1, fn1, tp2, tn2, fp2, fn2):
    confusion_matrix_1 = np.array(expand_counts_to_labels(tp1, fp1, fn1, tn1))
    confusion_matrix_2 = np.array(expand_counts_to_labels(tp2, fp2, fn2, tn2))

    kappa = sklearn.metrics.cohen_kappa_score(confusion_matrix_1, confusion_matrix_2)
    return round(kappa,4)

def cohen_kappa_from_counts(tp1, tn1, fp1, fn1, tp2, tn2, fp2, fn2):
    confusion_matrix_1 = np.array([tp1, fp1, fn1, tn1])
    confusion_matrix_2 = np.array([tp2, fp2, fn2, tn2])
    
    observed_agreement_count = np.sum(np.minimum(confusion_matrix_1, confusion_matrix_2))
    total_samples = np.sum(confusion_matrix_1)
    Po = observed_agreement_count / total_samples

    proportion_1 = confusion_matrix_1 / total_samples
    proportion_2 = confusion_matrix_2 / total_samples
    Pe = np.sum(proportion_1 * proportion_2)

    kappa = (Po - Pe) / (1 - Pe) if (1 - Pe) != 0 else 1
    return round(kappa,4)

for i in range(len(df_user)):
    # Extract the counts for each category in the current row
    tp1, tn1, fp1, fn1 = df_user.loc[i, ['TP', 'TN', 'FP', 'FN']]
    tp2, tn2, fp2, fn2 = df_ai.loc[i, ['TP', 'TN', 'FP', 'FN']]
    # Calculate Cohen's Kappa for this row
    kappa = cohen_kappa_from_counts(tp1, tn1, fp1, fn1, tp2, tn2, fp2, fn2)
    kappa_from_sk = cohen_kappa_using_sklearn(tp1, tn1, fp1, fn1, tp2, tn2, fp2, fn2)
    print(kappa, kappa_from_sk)
    # making sure both calculations match - either one of them is fine
    assert(kappa == kappa_from_sk)
    kappa_scores.append({'Threshold':df_user.loc[i,['Threshold']]['Threshold'], 'Cohen\'s Kappa Score': kappa})

results_path = '../../datasets/one-million-reddit-questions/kappa.csv'
results_path = os.path.join(current_dir,results_path)
with open(results_path,'w+') as output_file:
    fc = csv.DictWriter(output_file,fieldnames=kappa_scores[0].keys())
    fc.writeheader()
    fc.writerows(kappa_scores)
print(kappa_scores)
