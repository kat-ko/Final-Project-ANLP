import sys
import os
from rouge_metric import PyRouge
import xmltodict

sys.path.insert(0, "N:/Organisatorisches/Bereiche_Teams/ID/03_Studenten/Korte/Newsletter/Automatic Text Summarization/PreSumm_dev/dataset/scisumm_2018/")
BASE_DIR = "N:/Organisatorisches/Bereiche_Teams/ID/03_Studenten/Korte/Newsletter/Automatic Text Summarization/PreSumm_dev/dataset/scisumm_2018/"

articles = []
summaries = []



summary_file_dir = "N:/Organisatorisches/Bereiche_Teams/ID/03_Studenten/Korte/Newsletter/Automatic Text Summarization/PreSumm_dev/scisumm2018/summaries/"

result_file_dir = "N:/Organisatorisches/Bereiche_Teams/ID/03_Studenten/Korte/Newsletter/Automatic Text Summarization/PreSumm_dev/scisumm2018/results/"


# Name of the gold standard summary like "cnndm_summary_0.raw_src"
# Name of the inference result summary like "result_cnndm_paper_0_step-1.candidate"

# steps:
#Jedem result sein original summary zuordnen

# Aus result die Nummer raussuchen und zusammenpacken mit dem gold result 
# dann entsprechend der Nummer in neue Liste pacen

'''
Comparing Results to Gold Standard Summaries
'''


for filename in os.listdir(result_file_dir):
    #if(subfolder == "Reference_XML"):
    print("file", filename)
    temp = filename.replace("result_cnndm_paper_", "")
    filenumber_str = temp.replace("_step-1.candidate", "")
    try:
        filenumber = int(filenumber_str)
    except:
        print("ignoring gold_file")
    print("number", filenumber, "\n")


