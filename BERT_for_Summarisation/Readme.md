# How to handle Bert for Summarisation?

For all files: Change the BASE_DIR to the correct path on your computer.


## Scisumm 2018
### Data Preparation
* Go into the folder /dataset
* Unzip the files in /cnndm and extract them into the folder
* Set all paths correctly and run the cnndm_data_preparation.py script

### To run the inference
* Go inside the folder scr/ and open a console
* type the bash command calling the train_scisumm2018.py 

## CNNDM
### Data Preparation
* Go into the folder /dataset
* Set all paths correctly and run the scisumm2018_data_preparation.py script

### To run the inference
* Go inside the folder scr/ and open a console
* type the bash command calling the train_cnndm.py 

## BERT SOURCE:
* Most of the code to run the BERT Inference is from https://github.com/nlpyang/PreSumm/tree/dev. 
* The pretrained model which is fine-tuned for Summarisation Tasks can also be found under https://drive.google.com/file/d/1kKWoV0QCbeIuFt85beQgJ4v0lujaXobJ/view.

