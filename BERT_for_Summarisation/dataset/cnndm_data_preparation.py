
'''
Data Preparation
'''

import sys
import os
from rouge_metric import PyRouge
import json
import ast

sys.path.insert(0, "N:/Organisatorisches/Bereiche_Teams/ID/03_Studenten/Korte/Newsletter/Automatic Text Summarization/PreSumm_dev/dataset/cnndm/")
BASE_DIR = "N:/Organisatorisches/Bereiche_Teams/ID/03_Studenten/Korte/Newsletter/Automatic Text Summarization/PreSumm_dev/dataset/cnndm/"

files_raw = []

for folder in os.listdir(BASE_DIR):
    for filename in os.listdir(BASE_DIR + folder + "/"):
        print(filename)
        file_raw = open(os.path.join(BASE_DIR + folder + "/" + filename))
        files_raw.append(file_raw)


# Extract files from folders
# folders_nr: set number of folders to extract data from
folders_nr = 5
files_as_dict = []

for i in range(folders_nr):
    print("Folder", i, "from", folders_nr)
    files_as_dict.append(json.load(files_raw[i]))


# Keys to access for every paper
files_as_dict[0][0].keys()

# Acess the first article
files_as_dict[0][0]

# Acess the label of the first sentence of the first article
files_as_dict[0][0]['labels'][0]


'''
Produce Data Structure from xlm Data

Exceptions occur, if one of the xml-tags is empty
'''

papers = []

# Shuffle through all Folders
for i in range(len(files_as_dict)):
    print("Folder ", i, "of ", len(files_as_dict))
    # Shuffle through all Papers in the Folder
    for j in range(len(files_as_dict[i])):
        print("Paper ", j, "of ", len(files_as_dict[i]))

        paper = []

        text = []
        summary = []
        # Shuffle through all Sentences in the Paper
        for k in range(len(files_as_dict[i][j]['src'])):
            text.append(files_as_dict[i][j]['src'][k])
            
            # check label 
            if(files_as_dict[i][j]['labels'][k]):
                summary.append(files_as_dict[i][j]['src'][k])

        paper.insert(0, text)
        paper.insert(1, summary)
        
        papers.append(paper)


'''
Processing for use with bert
'''

bert_papers = []

for i in range(len(papers)):
    paper = []

    text = []
    for j in range(len(range(len(papers[i][0])))):
        text.append("[CLS] [SEP] " + ' '.join(papers[i][0][j]))
    
    text = ' '.join(text)
    
    summary = []
    for j in range(len(range(len(papers[i][1])))):
        summary.append("[CLS] [SEP] " + ' '.join(papers[i][1][j]))
        
    summary = ' '.join(summary)

    paper.insert(0, text)
    paper.insert(1, summary)

    bert_papers.append(paper)

file_dir = "N:/Organisatorisches/Bereiche_Teams/ID/03_Studenten/Korte/Newsletter/Automatic Text Summarization/PreSumm_dev/cnndm/papers/"
# Write text to files 
for i in range(len(bert_papers)):
    print("Writing paper in file no. ", i, "of", len(bert_papers))
    filename = file_dir + "cnndm_paper_" + str(i) + ".raw_src"

    f= open(filename, "w+", encoding="utf-8")

    f.write(bert_papers[i][0])

    f.close()

file_dir = "N:/Organisatorisches/Bereiche_Teams/ID/03_Studenten/Korte/Newsletter/Automatic Text Summarization/PreSumm_dev/cnndm/summaries/"
# Write summary to files 
for i in range(len(bert_papers)):
    print("Writing summary in file no. ", i, "of", len(bert_papers))
    filename = file_dir + "cnndm_summary_" + str(i) + ".raw_src"

    f= open(filename, "w+", encoding="utf-8")

    f.write(bert_papers[i][1])

    f.close()


result_dir = "N:/Organisatorisches/Bereiche_Teams/ID/03_Studenten/Korte/Newsletter/Automatic Text Summarization/PreSumm_dev/cnndm/results/"
for i in range(len(bert_papers)):
    print("Making result directory for paper no. ", i, "of", len(bert_papers))
    dirname = result_dir + "cnndm_result_" + str(i)

    os.makedirs(dirname)