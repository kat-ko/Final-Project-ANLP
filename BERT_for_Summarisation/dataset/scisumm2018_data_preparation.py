import sys
import os
from rouge_metric import PyRouge
import xmltodict

sys.path.insert(0, "N:/Organisatorisches/Bereiche_Teams/ID/03_Studenten/Korte/Newsletter/Automatic Text Summarization/PreSumm_dev/dataset/scisumm_2018/")
BASE_DIR = "N:/Organisatorisches/Bereiche_Teams/ID/03_Studenten/Korte/Newsletter/Automatic Text Summarization/PreSumm_dev/dataset/scisumm_2018/"

articles = []
summaries = []


for folder in os.listdir(BASE_DIR):
    for subfolder in os.listdir(BASE_DIR + folder):
        for filename in os.listdir(BASE_DIR + folder + "/" + subfolder):
            if(subfolder == "Reference_XML"):
                print("file", filename)
                xml_dict = xmltodict.parse(open(os.path.join(BASE_DIR + folder + "/" + subfolder + "/" + filename),encoding='utf-8', errors='ignore').read())
                articles.append(xml_dict)
            else:
                #print(os.path.join(BASE_DIR + folder + "/" + subfolder + "/" + filename))
                print("Summary", filename)
                summary = open(os.path.join(BASE_DIR + folder + "/" + subfolder + "/" + filename), encoding='utf-8', errors='ignore').read()
                summaries.append(summary)   



# Splitting the summary into sentences and removig empty string
summaries_split = []

for i in range(len(summaries)):
    summaries_split.insert(i, summaries[i].split("\n"))
    while True:
        try:
            summaries_split[i].remove("")
        except ValueError:
            break

papers = []

'''
Produce Data Structure from xlm Data

Exceptions occur, if one of the xml-tags is empty
'''
for i in range(len(articles)):
    paper = []
    for key in articles[0]['PAPER'].keys():
        if(key == 'S'):
            title = articles[0]['PAPER']['S']['#text']
            #append title to paper
            paper.insert(0, title)
        
        text_sentences = []

        # Shuffle through all sentences in the abstract
        if(key == 'ABSTRACT'):
            try:
                for j in range(len(articles[i]['PAPER']['ABSTRACT']['S'])):
                        sentence = articles[i]['PAPER']['ABSTRACT']['S'][j]['#text']
                        text_sentences.append(sentence)
            except:
                print("Exception in Abstract:", "i", i)


        # Shuffle through all sentences in the sections - introduction, methods, results, ...
        # articles[0]['PAPER']['SECTION'][0]['S'][0]['#text']
        if(key == 'SECTION'):
            try:
                for j in range(len(articles[i]['PAPER']['SECTION'])):
                        for k in range(len(articles[i]['PAPER']['SECTION'][j]['S'])):
                            sentence = articles[i]['PAPER']['SECTION'][j]['S'][k]['#text']
                            text_sentences.append(sentence)
            except:
                print("Exception in Text:", "i", i, "j", j)
            
        #append sentences to paper
        paper.insert(1, text_sentences)
        paper.insert(2, summaries_split[i])

    papers.append(paper)



'''
Processing for use with bert
'''

bert_papers = []

for i in range(len(papers)):
    paper = []

    text = []
    for j in range(len(range(len(papers[i][1])))):
        text.append("[CLS] [SEP] " + papers[i][1][j])
    
    text = ' '.join(text)
    
    summary = []
    for j in range(len(range(len(papers[i][2])))):
        summary.append("[CLS] [SEP] " + papers[i][2][j])
        
    summary = ' '.join(summary)

    paper.insert(0, text)
    paper.insert(1, summary)

    bert_papers.append(paper)



file_dir = "N:/Organisatorisches/Bereiche_Teams/ID/03_Studenten/Korte/Newsletter/Automatic Text Summarization/PreSumm_dev/scisumm2018/papers/"
# Write text to files 
for i in range(len(bert_papers)):
    print("Writing paper in file no. ", i, "of", len(bert_papers))
    filename = file_dir + "scisumm_paper_" + str(i) + ".raw_src"

    f= open(filename, "w+", encoding="utf-8")

    f.write(bert_papers[i][0])

    f.close()

file_dir = "N:/Organisatorisches/Bereiche_Teams/ID/03_Studenten/Korte/Newsletter/Automatic Text Summarization/PreSumm_dev/scisumm2018/summaries/"
# Write summary to files 
for i in range(len(bert_papers)):
    print("Writing summary in file no. ", i, "of", len(bert_papers))
    filename = file_dir + "scisumm_summary_" + str(i) + ".raw_src"

    f= open(filename, "w+", encoding="utf-8")

    f.write(bert_papers[i][1])

    f.close()
