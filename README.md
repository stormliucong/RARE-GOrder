# RARE-GOrder

## Description
This Project aims to develop a machine-learning model to recommend whether to use ES/GS or gene panels as the second-tier diagnostic test, based on the patient phenotypic manifestation documented in Electronic Health Records (EHRs). All essential codes used to construct the model along with codes supporting analysis results in the manuscript were entailed in this repository.

## Installation
1. Git clone this repository
2. Use the command below to install all required packages. 
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repeat Training Pipeline
### Data
- The original data is collected by the Clinical Genetic Division at Department of Pediatrics at Columbia Universtiy Irving Medical Center. Given the clinical data containing Protected Health Information (PHI) thus cannot be made readily available for public distirbution, we provided some [synthetic data](data_preprocessing/demo_data) for any reference to execute the model training pipeline.
- Examples of training a new model based on Columbia's dataset can be found in the [folder](model_pipeline/model_running_iterations/).
  
### Usage
To use `new_utils.py`, you have to create a local credential file `db.conf`. Keep it in a screte place with proper access management. Remember to fill in details in the {}.
```
[ELILEX]
server = {server_name}
ohdsi = {database_name}
preepicnotes = PreEpicNotes
username = {ohdsi_username}
password = {ohdsi_password}

[SOLR]
solrhost = {solr_url}
username = {solr_username}
password = {solr_password}
```

## For Prediction
### Data Preprocessing & Model Prediction
- The customized data preprocessor and trained model can be found in the [folder](analysis/saved_model/), along with feature mapping dictionary.
- Examples of how to use the model on your own dataset can be found in the [folder](analysis/saved_model/prediction.ipynb).

