# How to handle TF-IDF for Summarisation?

#### Steps to complete:
* Install the packages if any are missing (pip install *package_name*)
* Download the datasets of [Scisumm](https://github.com/WING-NUS/scisumm-corpus/tree/master/data/Training-Set-2018) and [CNN/Daily-Mail](https://drive.google.com/uc?id=1-DLTTioISS8i3UrOG4sjjc_js0ncnBnn)
* Adjust the paths in the beginning of the code to the folders with downloaded datasets
* Run the code

It takes around 15 minutes to run CNNDM file and around an hour for scisumm file. <br/>
It is recommended to start with CNNDM file as it loads the results faster and has a bit higher accuracy of the results.

#### About the files:
There are two files for running two different datasets:
* scisumm.ipynb file for a small dataset of the scientific papers 
* CNNDM.ipynb file for a bigger dataset of the news articles from CNN and Daily-Mail 

Both files consist of implementations TF-IDF and SMMRY algoritms and the code is very similar. 
There is also sklearn implementation of TF-IDF in the end of the files for comparison.


