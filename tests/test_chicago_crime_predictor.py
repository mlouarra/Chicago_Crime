# contenu de tests/test_chicago_crime_predictor.py
from pathlib import Path
import sys
import joblib
import pandas as pd
from prophet import Prophet
from pytest import fixture
sys.path.append(str(Path(__file__).parent.parent))
print(sys.path.append(str(Path(__file__).parent.parent)))
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.chicago_crime_predictor import ChicagoCrimePredictor

def test_init():
    predictor = ChicagoCrimePredictor(months_pred=6, data_dir='data')
    assert predictor._month_pred == 6
    assert predictor.data_dir == Path('data')
    assert predictor.model is None

@fixture
def mock_df_crime():
    return pd.DataFrame({
        'date': pd.to_datetime(['2015-03-18 12:00:00', '2018-12-20 15:00:00']),
        'primary_type': ['DECEPTIVE PRACTICE', 'ROBBERY'],
        'community_area': [32.0, 19.0]
    })
@fixture
def mock_df_socio():
    return pd.DataFrame({
            'community_area': [1.0, 2.0, 3.0, 4.0, 5.0],
            'community_area_name': ['Rogers Park', 'West Ridge', 'Uptown', 'Lincoln Square', 'North Center'],
        })

def test_load_df_crimes(mocker, mock_df_crime):


    mocker.patch('pandas.read_csv', return_value=mock_df_crime)
    predictor = ChicagoCrimePredictor(months_pred=6, data_dir='data')
    df = predictor.load_df_crimes()
    assert df.equals(mock_df_crime)

def test_load_df_socio(mocker, mock_df_socio):
    mocker.patch('pandas.read_csv', return_value=mock_df_socio)
    predictor = ChicagoCrimePredictor(months_pred=6, data_dir='data')
    df = predictor.load_df_socio()
    assert df.equals(mock_df_socio)

def test_return_data(mocker, mock_df_crime, mock_df_socio):
    mocker.patch.object(ChicagoCrimePredictor, 'load_df_crimes', return_value=mock_df_crime)
    mocker.patch.object(ChicagoCrimePredictor, 'load_df_socio', return_value=mock_df_socio)

    predictor = ChicagoCrimePredictor(months_pred=6, data_dir='data')
    train, test = predictor.return_data(type_incident='THEFT')
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)


def test_model_train(mocker):
    mocker.patch.object(Prophet, 'fit')
    predictor = ChicagoCrimePredictor(months_pred=6, data_dir='data')
    predictor.model_train(pd.DataFrame({'ds': [1], 'y': [1]}))
    assert isinstance(predictor.model, Prophet)


def test_model_predict(mocker):
    mocker.patch.object(Prophet, 'predict', return_value=pd.DataFrame({'ds': [1], 'yhat': [1]}))
    predictor = ChicagoCrimePredictor(months_pred=6, data_dir='data')
    predictor.model = Prophet()
    forecast, predictions = predictor.model_predict()
    assert not predictions.empty


def test_model_predict():
    # Chemin vers le modèle pré-entraîné
    model_path = Path('./models')

    # Charger le modèle pré-entraîné
    trained_model = joblib.load(model_path)

    # Créer une instance de votre prédicteur
    predictor = ChicagoCrimePredictor(months_pred=6, data_dir='data')

    # Attribuer le modèle pré-entraîné à l'instance de prédicteur
    predictor.model = trained_model

    # Appeler la méthode model_predict
    forecast, predictions = predictor.model_predict()

    # Assertions pour votre test
    assert not predictions.empty
    # Vous pouvez ajouter plus d'assertions ici pour valider les résultats
