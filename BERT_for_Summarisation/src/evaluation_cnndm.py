import sys
import os
from rouge import Rouge 
import numpy as np

sys.path.insert(0, "N:/Organisatorisches/Bereiche_Teams/ID/03_Studenten/Korte/Newsletter/Automatic Text Summarization/PreSumm_dev/cnndm/")
BASE_DIR = "N:/Organisatorisches/Bereiche_Teams/ID/03_Studenten/Korte/Newsletter/Automatic Text Summarization/PreSumm_dev/cnndm/"

summary_file_dir = "N:/Organisatorisches/Bereiche_Teams/ID/03_Studenten/Korte/Newsletter/Automatic Text Summarization/PreSumm_dev/cnndm/summaries/"
result_file_dir = "N:/Organisatorisches/Bereiche_Teams/ID/03_Studenten/Korte/Newsletter/Automatic Text Summarization/PreSumm_dev/cnndm/results/"

'''
Comparing Results to Gold Standard Summaries
'''

results = []

for filename in os.listdir(result_file_dir):
    result = []
    #if(subfolder == "Reference_XML"):
    print("file", filename)
    temp = filename.replace("result_cnndm_paper_", "")
    filenumber_str = temp.replace("_step-1.candidate", "")
    try:
        filenumber = int(filenumber_str)
        print("number", filenumber, "\n")

        result.append(filenumber)

        filesize = os.path.getsize(result_file_dir + filename)
        if filesize > 0:
            text = open(result_file_dir + filename, encoding="utf-8").read()
            result.append(text)

            results.append(result)
    except:
        print("ignoring gold_file")


summaries = []
for filename in os.listdir(summary_file_dir):
    summary = []
    #if(subfolder == "Reference_XML"):
    print("file", filename)
    temp = filename.replace("cnndm_summary_", "")
    filenumber_str = temp.replace(".raw_src", "")
        
    filenumber = int(filenumber_str)
    print("number", filenumber, "\n")

    summary.append(filenumber)

    raw = open(summary_file_dir + filename, encoding="utf-8").read()
    text = raw.replace("[CLS] [SEP] ", "")
    summary.append(text)

    summaries.append(summary)

'''
Now put them together into one Datastructure
'''
# Function to sort the results by number 
def sortFunc(list_element):
    number = list_element[0]
    return number

results.sort(key=sortFunc)

# Adding the gold summaries to the data structure
for i in range(len(results)):
    for j in range(len(summaries)):
        if(summaries[j][0] == results[i][0]):
            results[i].append(summaries[j][1])

print(results[0])



'''
Use Rouge metric to compare the summarisations then write Scores to files
'''

scores = []

rouge = Rouge()

from rouge import Rouge 

score_dir = BASE_DIR + "scores/"

f_all = open(score_dir + "all_scores_cnndm.txt", "w+", encoding="utf-8")
f_mean = open(score_dir + "mean_scores_cnndm.txt", "w+", encoding="utf-8")

# One for every Summarisation
# (makes it easier to evaluate manually)
for i in range(len(results)):

    print("Evaluation file no.", results[i][0])
    
    filename = score_dir + "scores_" + str(results[i][0]) + ".txt"
    #f = open(filename, "w+", encoding="utf-8")

    try: 
        hypothesis = results[i][1]
        reference = results[i][2]

        score = rouge.get_scores(hypothesis, reference)
        scores.append(score)

        string = "BERT Result: " + hypothesis + "\n" + " Gold Summary: " + reference + "\n" + " Scores", score

        #f.write(str(string))
        #f.close()

    except:
        print("For file nr.", results[i][0], "an Exception ocurred")

print("Writing all scores")
f_all.write(str(scores))

smmry_rouge_1_r = list(); smmry_rouge_1_p = list(); smmry_rouge_1_f = list()
smmry_rouge_2_r = list(); smmry_rouge_2_p = list(); smmry_rouge_2_f = list()
smmry_rouge_l_r = list(); smmry_rouge_l_p = list(); smmry_rouge_l_f = list()

for i in range(len(scores)): 
    smmry_rouge_1_r.append(scores[i][0]['rouge-1']['r'])
    smmry_rouge_2_r.append(scores[i][0]['rouge-2']['r'])
    smmry_rouge_l_r.append(scores[i][0]['rouge-l']['r'])
    
    smmry_rouge_1_p.append(scores[i][0]['rouge-1']['p'])
    smmry_rouge_2_p.append(scores[i][0]['rouge-2']['p'])
    smmry_rouge_l_p.append(scores[i][0]['rouge-l']['p'])
    
    smmry_rouge_1_f.append(scores[i][0]['rouge-1']['f'])
    smmry_rouge_2_f.append(scores[i][0]['rouge-2']['f'])
    smmry_rouge_l_f.append(scores[i][0]['rouge-l']['f'])
      
string1 = "Mean rouge 1 score: " + "r:" + str(np.mean(smmry_rouge_1_r)) + " p:" + str(np.mean(smmry_rouge_1_r)) + "f:" + str(np.mean(smmry_rouge_1_f)) + "\n" 
string2 = "Mean rouge 2 score: " + "r:" + str(np.mean(smmry_rouge_2_r)) + " p:" + str(np.mean(smmry_rouge_2_r)) + "f:", str(np.mean(smmry_rouge_2_f)) + "\n" 
stringl = "Mean rouge L score: " + "r:" + str(np.mean(smmry_rouge_l_r)) + " p:" + str(np.mean(smmry_rouge_l_r)) + "f:", str(np.mean(smmry_rouge_l_f)) + "\n" 

mean_scores = str(string1) + str(string2) + str(stringl)

print("Writing mean scores")
f_mean.write(mean_scores)