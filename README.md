# ANLP Final Project

<b>Automatic Summarisation of scientific papers</b>

In the following chapters, we describe what is inside of this repository and how to run the code.


### The Datasets Used: 
* Scisumm 2018 (accessible on [Github](https://github.com/WING-NUS/scisumm-corpus/tree/master/data/Training-Set-2018)).  This dataset is small, it consists of 30 papers and their extractive target summarisations.
* CNN/Daily-Mail (accessible on: [Google Drive](https://drive.google.com/uc?id=1-DLTTioISS8i3UrOG4sjjc_js0ncnBnn)). More information about the dataset can be found [here](https://www.tensorflow.org/datasets/catalog/cnn_dailymail). This dataset contains around 30.000 articles of multiple subjects together with the extractive target summarisations for each article.

### Machine Learning Methods:
* [TF-IDF](https://github.com/kat-ko/Final-Project-ANLP/tree/main/TF-IDF_for_Summarisation) (Trequency–Inverse Document Frequency) implemented ourselves and also taken from scikit learn library for comparison. 
* [SMMRY](https://github.com/kat-ko/Final-Project-ANLP/tree/main/TF-IDF_for_Summarisation). To  implement SMMRY, TF-IDF was improved by using part-of-speech tagging and TextRank algoritm.  
* Pre-trained [BERT](https://github.com/kat-ko/Final-Project-ANLP/tree/main/BERT_for_Summarisation) model
