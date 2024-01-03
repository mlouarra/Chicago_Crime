.
├── config

│── config.yml

└── rename_columns.json

├── data 
    

├── docs

├── LICENSE

├── main_api.py

├── main.py

├── Makefile

├── models

├── notebooks

├── README.md

├── reports

├── requirements.txt

├── setup.py

├── src

├── test_environment.py

├── tests

└── tox.ini

# Projet sur les Crimes à Chicago

## Description
Ce projet analyse les données sur les crimes à Chicago dans 
le but de prédire les tendances criminelles et de fournir 
des insights pour la prévention. 
Il utilise des données historiques pour identifier les modèles et 
les hotspots de criminalité.
## source des données 
- https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/data_preview
- https://data.cityofchicago.org/Health-Human-Services/Census-Data-Selected-socioeconomic-indicators-in-C/kn9c-c2s2/data_preview

## Installation
### Prérequis
- Python 3.9
- bibliothèque pandas
- bibliothèque scikit-learn
- bibliothèque Prophet

### Setup
```sh
git clone git@github.com:mlouarra/Chicago_Crime.git
cd projet_crime_chicago
pip install -r requirements.txt

