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

        df_crimes = pd.read_csv("../data/raw/Crimes_Chicago.csv", usecols=['Date', 'Primary Type', 'Community Area'], parse_dates=['Date'])
        df_crimes.rename(columns=self.dicto_rename_crimes, inplace=True)
        return df_crimes

    def load_df_socio(self):

        df_socio = pd.read_csv("../data/raw/socio_economic_Chicago.csv")
        df_socio.rename(columns=self.dicto_rename_socio, inplace=True)
        return df_socio


    def return_data_ml(self, type_incident, start_date_train, end_date_train, community_area=None):
        # Conversion des chaînes de dates en objets datetime.
        start_date_train, end_date_train = pd.to_datetime([start_date_train, end_date_train])

        # Chargement des DataFrames.
        df_crimes = self.load_df_crimes()
        df_socio = self.load_df_socio()

        # Fusion des DataFrames sur 'community_aa_number' avec une jointure gauche.
        df = pd.merge(df_crimes, df_socio[['community_area_number', 'community_area_name']],
                      on='community_area_number', how='left')

        # Filtrage des crimes par type et, si spécifié, par community_area.
        is_type = df['primary_type'] == type_incident
        is_area = df['community_area_name'] == community_area if community_area else True
        df = df[is_type & is_area]

        # Ajout d'une colonne 'year_month' pour le groupement.
        df['year_month'] = df['date'].dt.to_period('M')

        # Groupement par 'year_month' et comptage des incidents.
        df_group = df.groupby('year_month')['primary_type'].count().reset_index()

        # Renommage des colonnes pour la conformité avec Prophet.
        df_group.rename(columns={'year_month': 'ds', 'primary_type': 'y'}, inplace=True)

        # Conversion de la colonne 'ds' en fin de mois et filtrage par la plage de dates.
        df_group['ds'] = df_group['ds'].dt.to_timestamp('M') + MonthEnd(1)

        # Retour des données filtrées par la plage de dates d'entraînement.
        return df_group[(df_group['ds'] >= start_date_train) & (df_group['ds'] <= end_date_train)]


    def split_date(self, months, df):
        df['ds'] = pd.to_datetime(df['ds'])  # Assurez-vous que 'ds' est au format datetime
        split_date = df['ds'].max() - pd.DateOffset(months=months)
        return split_date

    def model_train(self, data_train, split_date):
        # Séparer les données d'entraînement et de test

        train = data_train[data_train['ds'] <= split_date]
        # Créer et entraîner le modèle Prophet
        self.model = Prophet(yearly_seasonality=True,
                        # weekly_seasonality=True,
                        daily_seasonality=False,
                        seasonality_prior_scale=10,  # Augmenter pour une saisonnalité plus flexible
                        changepoint_prior_scale=0.05  # Diminuer pour des tendances moins flexibles)
                        )
        self.model.fit(train)



    def model_predict(self):
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné. Utilisez la méthode 'model_train' d'abord.")

        # Prédire les valeurs pour les dates futures
        future = self.model.make_future_dataframe(periods=12, freq='M')
        forecast = self.model.predict(future)
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
    data_ml = obj_predict.return_data_ml_("THEFT", "2019-11", "2022-10", "Austin")
    # obj_predict.model_train(data_ml)
    # obj_predict.model_save("../models/model_theft")
    
    print(data_ml)
