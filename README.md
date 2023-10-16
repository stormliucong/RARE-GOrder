# RARE-GOrder

## Description
This is the project aims to provide an automated system for genetic testing order include gene panels, WGS/WES sequencing. All essential codes used to construct the model along with codes supporting analysis results in the manuscript were entailed in this repository.


## Table of Contents

- [Installation](#installation)
- [Data](#data)
- 

## Installation

Use the command below to install all required packages. 
```
pip install -r requirements.txt
        OR
pip3 install -r requirements.txt
```

## Data

The original data is collected by the Clinical Genetic Division at Department of Pediatrics at Columbia Universtiy Irving Medical Center. Given the clinical data containing Protected Health Information (PHI) thus cannot be made readily available for public distirbution, we provided some [synthetic data](data_preprocessing/demo_data) for any reference to execute the model training pipeline.


## Usage
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



