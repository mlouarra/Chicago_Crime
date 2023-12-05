import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from prophet.serialize import model_to_json, model_from_json
from prophet import Prophet
import joblib
from pandas.tseries.offsets import MonthEnd
import joblib

class ChicagoCrimePredictor:
    dicto_rename_crimes = {
        'ID': 'id',
        'Case Number': 'cas_number',
        'Date': 'date',
        'Block': 'block',
        'IUCR': 'iucr',
        'Primary Type': 'primary_type',
        'Description': 'description',
        'Location Description': 'location_description',
        'Arrest': 'arrest',
        'Domestic': 'domestic',
        'Beat': 'beat',
        'District': 'district',
        'Ward': 'ward',
        'Community Area': 'community_area_number',
        'FBI Code': 'fbi_code',
        'X Coordinate': 'x_coordinate',
        'Y Coordinate': 'y_coordinate',
        'Year': 'year',
        'Updated On': 'updated_on',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Location': 'location'
    }

    dicto_rename_socio = {
        'Community Area Number':'community_area_number',
        'COMMUNITY AREA NAME':'community_area_name',
        'PERCENT OF HOUSING CROWDED':'pct_housing_crowded',
        'PERCENT HOUSEHOLDS BELOW POVERTY':'pct_households_below_poverty',
        'PERCENT AGED 16+ UNEMPLOYED':'pct_age16_unemployed',
        'PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA':'pct_age25_no_highschool',
        'PERCENT AGED UNDER 18 OR OVER 64': 'pct_not_working_age',
        'per_capita_income':'per_capita_income',
        'HARDSHIP INDEX' : 'hardship_index'}


    def __init__(self):
        self.model = None


    def load_df_crimes(self):

        df_crimes = pd.read_csv("../data/raw/Crimes_-_2001_to_Present_20231130.csv", parse_dates=['Date'])
        df_crimes.rename(columns=self.dicto_rename_crimes, inplace=True)
        return df_crimes

    def load_df_socio(self):

        df_socio = pd.read_csv("../data/raw/Census_Data_-_Selected_socioeconomic_indicators_in_Chicago__2008___2012.csv")
        df_socio.rename(columns=self.dicto_rename_socio, inplace=True)
        return df_socio

    def return_data_ml(self, type_incident, start_date_train, end_date_train, community_area=None):

        start_date_train = pd.to_datetime(start_date_train)
        end_date_train = pd.to_datetime(end_date_train)
        df_crimes = self.load_df_crimes()
        df_socio = self.load_df_socio()
        df_src = pd.merge(df_crimes, df_socio, on='community_area_number', how='left')

        df_src.drop(df_src.columns.difference(['primary_type', 'date', 'community_area_name']), inplace=True, axis=1)
        if community_area != None:
            df = df_src[(df_src.primary_type == type_incident) & (df_src.community_area_name == community_area)]
        else:
            df = df_crimes[df_crimes.primary_type == type_incident]
        del df_crimes
        del df_socio
        df['year_month'] = df['date'].apply(lambda x: x.strftime('%Y-%m'))
        df_group = df.groupby(['year_month'], as_index=False).agg({'primary_type': 'count'})
        df_group.rename(columns={"primary_type": "nb_crime"}, inplace=True)
        df_group['year_month'] = pd.to_datetime(df_group['year_month'])
        df_group.sort_values(by='year_month', inplace=True)
        df_group.reset_index(inplace=True, drop=True)
        del df
        df_group['year_month'] = pd.to_datetime(df_group['year_month'], format="%Y%m") + MonthEnd(1)
        df_group.columns = ['ds', 'y']
        return df_group[(df_group['ds'] >= start_date_train) & (df_group['ds'] <= end_date_train)]


    def model_train(self, data_train):

        # Créer un modèle Prophet
        self.model = Prophet()

        # Entraîner le modèle sur les données
        self.model.fit(data_train)

    def model_predict(self, future_dates):
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné. Utilisez la méthode 'model_train' d'abord.")

        # Prédire les valeurs pour les dates futures
        forecast = self.model.predict(future_dates)

        return forecast

    def model_save(self, filename):
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné. Utilisez la méthode 'model_train' d'abord.")

        # Enregistrer le modèle dans un fichier
        joblib.dump(self.model, filename)
        print(f"Modèle enregistré sous {filename}")

    def model_evaluation(self, actual_values, predicted_values):
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné. Utilisez la méthode 'model_train' d'abord.")

        # Calcul de la MAE
        mae = mean_absolute_error(actual_values, predicted_values)

        # Calcul du RMSE
        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))

        # Affichage des métriques d'évaluation
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")

# Exemple d'utilisation :
if __name__ == "__main__":
    obj_predict = ChicagoCrimePredictor()
    data_ml = obj_predict.return_data_ml("THEFT", "2019-11", "2022-10", "Austin")
    obj_predict.model_train(data_ml)
    obj_predict.model_save("../models/model_theft")
    
    print(data_ml)
