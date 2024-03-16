import plotly.io as pio
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
from src import chicago_crime_predictor as ccp

# Chemins relatifs basés sur le chemin de base
DATA_DIR = BASE_DIR / 'data'

obj_predict = ccp.ChicagoCrimePredictor(months_pred=12, data_dir=DATA_DIR)
def visualize_results(df_train, df_test, predictions):
    fig = obj_predict.model_visualization(df_train, df_test, predictions)
    tmp_file_path = "model_visualization.html"
    pio.write_html(fig, tmp_file_path)
    return tmp_file_path
