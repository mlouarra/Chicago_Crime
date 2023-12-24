# contenu de tests/test_chicago_crime_predictor.py
import pytest
from src.chicago_crime_predictor import ChicagoCrimePredictor
import pandas as pd
from pytest import fixture
from pathlib import Path

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
        'community_area_number': [32.0, 19.0]
    })
@fixture
def mock_df_socio():
    return pd.DataFrame({
            'community_area_number': [1.0, 2.0, 3.0, 4.0, 5.0],
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
