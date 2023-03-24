# .ONESHELL runs everything in a single-shells (instead of multiple subshells as is standard with make)
# Needed for using commands like CONDA_ACTIVATE that build off each other
.ONESHELL:

CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate; conda activate
CONDA_DEACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda deactivate; conda deactivate

# phony targets don't have directly associated file that is created
.PHONY: all conda_venv env_packages activate_venv delete_conda_env clean 

# run whole pipeline
all: data\raw\iris.csv \
	data\featurized\featured_iris.csv \
	data\split\train.csv data\split\test.csv \
	models\model.joblib \
	reports\confusion_matrix.png reports\metrics.json
	
# ------- INDIVIDUAL PIPELINE STAGES ----------------------
# get raw data from remote 
# To run: 
# 	> make data\raw\iris.csv
data\raw\iris.csv: src\stages\data_load.py ..\remote_iris\iris.csv
	python src\stages\data_load.py -i ..\remote_iris\iris.csv -o data\raw\iris.csv

# featurize raw data
# To run: 
#   > make data\featurized\featured_iris.csv
data\featurized\featured_iris.csv: src\stages\featurize.py data\raw\iris.csv
	python src\stages\featurize.py -i data\raw\iris.csv -o data\featurized\featured_iris.csv

# split featurized data
# To run:
#  > make data\split\train.csv data\split\test.csv
data\split\train.csv data\split\test.csv: src\stages\data_split.py data\featurized\featured_iris.csv
	python src\stages\data_split.py -i data\featurized\featured_iris.csv -o data\split\train.csv data\split\test.csv

#-------------- ALTERNATIVE WAY TO WRITE MULTIPLE TARGETS ------------------------------------------------------------
# data\split\train.csv: src\stages\data_split.py data\featurized\featured_iris.csv
# 	python src\stages\data_split.py -i data\featurized\featured_iris.csv -o data\split\train.csv data\split\test.csv

# data\split\test.csv: src\stages\data_split.py data\featurized\featured_iris.csv
# 	python src\stages\data_split.py -i data\featurized\featured_iris.csv -o data\split\train.csv data\split\test.csv
#----------------------------------------------------------------------------------------------------------------------

# train classifier model
# To run:
#  > make models\model.joblib
models\model.joblib: src\stages\train.py data\split\train.csv
	python src\stages\train.py -i data\split\train.csv -o models\model.joblib

# evaluate classifier
# To run:
#  > make reports\confusion_matrix.png reports\metrics.json
reports\confusion_matrix.png reports\metrics.json: src\stages\evaluate.py models\model.joblib data\split\test.csv
	python src\stages\evaluate.py -i models\model.joblib data\split\test.csv -o reports\metrics.json reports\confusion_matrix.png

# ----------- END Pipeline Stages ----------


# install packages for project into virtual environment
# also install source code as src package into virtual environment
env_packages: 
	pip install -r requirements.txt
	pip install -e .  

# create conda environment named "make_demo" and activate it
conda_venv:
	conda create --name make_demo python=3.9.5
	$(CONDA_ACTIVATE) make_demo
	pip install -r requirements.txt
	pip install -e .  

delete_conda_env:
	$(CONDA_DEACTIVATE) 
	conda env remove --name make_demo

# deleting all pipeline outputs
clean:
	rm -f data\raw\iris.csv \
	rm -f data\featurized\featured_iris.csv \
	rm -f data\split\train.csv \
	rm -f data\split\test.csv \
	rm -f models\model.joblib \
	rm -f reports\confusion_matrix.png \
	rm -f reports\metrics.json