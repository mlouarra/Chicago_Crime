import pandas as pd
from pathlib import Path

# Chemins relatifs basés sur le chemin de base
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from src import chicago_crime_predictor as ccp

DATA_DIR = BASE_DIR / 'data'

def preprocess_data(data_dir):
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / data_dir

    obj_predict = ccp.ChicagoCrimePredictor(months_pred=12, data_dir=DATA_DIR)
    obj_predict.df_process()
    df_train, df_test = obj_predict.return_data("ASSAULT", 'Austin')

    return df_train, df_test
