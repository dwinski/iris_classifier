# Iris Classification Demo Project
Project to demonstrate use of command-line arguments and makefile to run a machine learning pipeline

## Description
Project uses a series of scripts to train a classifier model on the famous iris dataset.

Individual stages of the pipeline are in src/stages folder and sequenced as follows:
1. data_load.py (loading and storing raw data from external source)
2. featurize.py (featurizing raw data and storing featurized data)
3. data_split.py (split featurized data into train and test sets and store them)
4. train.py (train the classifier model on the training set and store the model)
5. evaluate.py (evaluate the performance of the model on the test set and store the metrics)

## Virtual Environment
A conda environment was used running Python 3.9.5 with packages installed via pip install of
the requirements.txt file in the project folder.  The direct dependencies of the project
environment are listed in the requirements.in file.

## Running Project
Project can be run from end-to-end via makefile command run from project folder using:
> make all

Individual steps of the pipeline can be run as indicated in the makefile.
All project artifacts/outputs can be deleted using:
> make clean

More detail on inputs/outputs and usage of stages can be obtained by running:
> python src\stages\\<name_of_.py_file> --help

e.g.
> python src\stages\train.py --help

## Author/Contact
David Winski (David.Winski@va.gov)





