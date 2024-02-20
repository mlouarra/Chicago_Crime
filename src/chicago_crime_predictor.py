import joblib
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import sqlite3
from datetime import datetime
import requests

# Importez la classe Logger
from src.logger import Logger

# Configurez le logger
logger = Logger('log.txt').get_logger()


class ChicagoCrimePredictor:

    """
    Cette classe est utilisée pour prédire les crimes à Chicago en utilisant des données historiques
    et socio-économiques. Elle implémente les fonctionnalités pour charger des données, entraîner un modèle
    prédictif, faire des prédictions, sauvegarder le modèle, évaluer les performances et visualiser les résultats.
    """
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
        'Community Area Number': 'community_area',
        'COMMUNITY AREA NAME': 'community_area_name',
        'PERCENT OF HOUSING CROWDED': 'pct_housing_crowded',
        'PERCENT HOUSEHOLDS BELOW POVERTY': 'pct_households_below_poverty',
        'PERCENT AGED 16+ UNEMPLOYED': 'pct_age16_unemployed',
        'PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA': 'pct_age25_no_highschool',
        'PERCENT AGED UNDER 18 OR OVER 64': 'pct_not_working_age',
        'per_capita_income': 'per_capita_income',
        'HARDSHIP INDEX': 'hardship_index'}

    def __init__(self, months_pred, data_dir):

        """
        Initialise l'instance de ChicagoCrimePredictor.

        :param months_pred: Nombre de mois pour les prédictions futures.
        :param data_dir: Chemin du répertoire contenant les fichiers de données.
        """
        self.model = None
        self._month_pred = months_pred
        self.data_dir = Path(data_dir)
        self.path_raw = self.data_dir / 'raw'
        self.path_process = self.data_dir/ 'processed'

    def update_crime_data(self):
        """
        Met à jour les données sur les crimes en récupérant les dernières entrées depuis l'API.
        """
        logger.info("Début de la mise à jour des données sur les crimes depuis l'API.")
        # Charger le DataFrame existant
        logger.info("lecture du fichier Crime_Chicago.csv")
        df_crimes = pd.read_csv(self.path_process.joinpath("Crimes_Chicago.csv"))
        # Trouver la date la plus récente dans le fichier CSV
        last_date = df_crimes['date'].max()
        logger.info(f"la date max est {last_date}")
        last_date = pd.to_datetime(last_date)
        start_date = last_date + pd.Timedelta(days=1)  # Commencer le jour suivant la d

        # Convertir start_date au format attendu par l'API (YYYY-MM-DD)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = datetime.now().strftime('%Y-%m-%d')  # Utiliser la date actuelle comme date de fin

        # Préparer l'URL et les paramètres de la requête
        base_url = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"
        params = {
            "$where": f"date >= '{start_date_str}' AND date <= '{end_date_str}'",
            "$select": "date, community_area, primary_type",
            "$limit": 500000  # Limite adaptée pour votre cas d'utilisation
        }

        # Effectuer la requête à l'API
        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            logger.info(f"Données mises à jour avec succès jusqu'au {end_date_str}.")
            # Charger les nouvelles données dans un DataFrame
            new_data = pd.DataFrame(response.json())
        else:
            logger.error(f"Erreur lors de la récupération des données depuis l'API. Code d'erreur : {response.status_code}")

        # Renommer les colonnes selon votre dictionnaire
        new_data.rename(columns=self.dicto_rename_crimes, inplace=True)

        # Convertir les colonnes de date au format datetime
        new_data['date'] = pd.to_datetime(new_data['date'])

        # Concaténer avec les anciennes données
        updated_df = pd.concat([df_crimes, new_data], ignore_index=True)

        # Sauvegarder le DataFrame mis à jour dans le fichier CSV
        updated_df.to_csv(self.path_process.joinpath( "Crimes_Chicago.csv"), index=False)



    def df_process(self):
        """

        Returns:

        """

        crimes_file_path = self.path_raw.joinpath('Crimes_Chicago.csv')
        df_crimes = pd.read_csv(crimes_file_path, usecols=['Date', 'Primary Type', 'Community Area'], parse_dates=['Date'], low_memory=False)
        df_crimes.rename(columns={"Date": "date", "Primary Type": "primary_type", 'Community Area': "community_area"}, inplace=True)
        df_crimes['community_area'] = df_crimes['community_area'].astype('Int64')
        df_crimes.to_csv(self.path_process.joinpath("Crimes_Chicago.csv"), index=False)


    def load_df_crimes(self):


        """

        Charge les données sur les crimes depuis un fichier CSV.


        :return: DataFrame contenant les données sur les crimes.
        """

        crimes_file_path = self.path_process.joinpath('Crimes_Chicago.csv')
        df_crimes = pd.read_csv(crimes_file_path, parse_dates=['date'], low_memory=False)

        return df_crimes

    def load_df_socio(self):
        """

        Charge les données socio-économiques depuis un fichier CSV.

        :return: DataFrame contenant les données socio-économiques.
        """
        socio_file_path = self.path_raw.joinpath('socio_economic_Chicago.csv')
        df_socio = pd.read_csv(socio_file_path)
        df_socio.rename(columns=self.dicto_rename_socio, inplace=True)
        return df_socio

    def return_data(self, type_incident, community_area=None):

        """

         Prépare et retourne les données pour l'entraînement et le test du modèle.

        :param type_incident: Type de crime à analyser.
        :param community_area: Zone communautaire spécifique à filtrer (optionnel).
        :return: Deux DataFrames - données d'entraînement et de test.
        """
        # Conversion des chaînes de dates en objets datetime.
        df_crimes = self.load_df_crimes()
        df_crimes['community_area'] = df_crimes['community_area'].astype('Int64')
        df_socio = self.load_df_socio()
        df_socio['community_area'] = df_socio['community_area'].astype('Int64')
        # Fusion des DataFrames sur 'community_area_number' avec une jointure gauche.
        df = pd.merge(df_crimes, df_socio[['community_area', 'community_area_name']],
                      on='community_area', how='left')
        df.sort_values(by="date", inplace=True)

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
        df_group['ds'] = df_group['ds'].dt.to_timestamp('M')
        # Retour des données filtrées par la plage de dates d'entraînement.
        df_group['ds'] = pd.to_datetime(df_group['ds'])  # Assurez-vous que 'ds' est au format datetime
        split_date = df_group['ds'].max() - pd.DateOffset(months=self._month_pred)
        train = df_group[df_group['ds'] <= split_date]
        test = df_group[df_group['ds'] > split_date]
        return train, test

    def model_train(self, data_train):
        """

         Entraîne le modèle Prophet sur les données fournies.

        :param data_train: DataFrame contenant les données d'entraînement.
        """

        # Créer et entraîner le modèle Prophet
        self.model = Prophet(yearly_seasonality=True,
                        # weekly_seasonality=True,
                        daily_seasonality=False,
                        seasonality_prior_scale=10,  # Augmenter pour une saisonnalité plus flexible
                        changepoint_prior_scale=0.05  # Diminuer pour des tendances moins flexibles)
                        )
        self.model.fit(data_train)


    def model_predict(self):

        """
        Effectue des prédictions en utilisant le modèle entraîné.

        :return: DataFrame contenant les prévisions et un DataFrame pour les prédictions après la date de séparation.
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné. Utilisez la méthode 'model_train' d'abord.")

        # Prédire les valeurs pour les dates futures
        future = self.model.make_future_dataframe(periods=self._month_pred, freq='M')
        forecast = self.model.predict(future)
        split_date = forecast['ds'].max() - pd.DateOffset(months=self._month_pred)
        predictions = forecast[['ds', 'yhat']].loc[forecast['ds'] > split_date]
        return forecast, predictions

    def model_save(self, filename):

        """

         Sauvegarde le modèle entraîné dans un fichier.

        :param filename: Nom du fichier dans lequel sauvegarder le modèle.
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné. Utilisez la méthode 'model_train' d'abord.")

        # Enregistrer le modèle dans un fichier
        joblib.dump(self.model, filename)
        print(f"Modèle enregistré sous {filename}")


    def model_load(self, filename):
        """
        Charge le modèle entraîné à partir d'un fichier .pkl.

        :param filename: Nom du fichier à partir duquel charger le modèle.
        :return: Le modèle chargé.
        """
        self.model = joblib.load(filename)
        print(f"Modèle chargé depuis {filename}")

    def model_evaluation(self, test, predictions):
        """

        Évalue le modèle sur les données de test.
        :param test: DataFrame contenant les données de test.
        :param predictions: DataFrame contenant les prédictions du modèle.
        """

        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné. Utilisez la méthode 'model_train' d'abord.")
        # Calcul de la MAE
        mae = mean_absolute_error(test['y'], predictions['yhat'])
        # Calcul du RMSE
        rmse = np.sqrt(mean_squared_error(test['y'], predictions['yhat']))
        # Calcul du R²
        r2 = r2_score(test['y'], predictions['yhat'])
        # Enregistrement dans la base de données
        connection = sqlite3.connect('./db/model_evaluation.db')
        cursor = connection.cursor()

        # Obtention de la date et l'heure actuelles
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Préparation de la commande SQL
        insert_query = '''INSERT INTO model_evaluation (mae, rmse, r2, date)
                                 VALUES (?, ?, ?, ?)'''
        data = (mae, rmse, r2, current_date)

        # Exécution de la commande
        cursor.execute(insert_query, data)

        # Sauvegarde des modifications et fermeture de la connexion
        connection.commit()
        connection.close()

        # Affichage des métriques d'évaluation
        logger.info(f"Évaluation du modèle terminée. MAE: {mae}, RMSE: {rmse}, R²: {r2}")

        return mae, rmse, r2

    def model_visualization(self, train, test, predictions):
        """

        Visualise les données d'entraînement, de test et les prédictions.

        :param train: DataFrame contenant les données d'entraînement.
        :param test: DataFrame contenant les données de test.
        :param predictions: DataFrame contenant les prédictions du modèle.
        """
        # Création des tracés pour Plotly
        trace_train = go.Scatter(x=train['ds'], y=train['y'], mode='lines', name='Training Data')
        trace_test = go.Scatter(x=test['ds'], y=test['y'], mode='lines', name='Test Data')
        trace_predictions = go.Scatter(x=predictions['ds'], y=predictions['yhat'], mode='lines', name='Predicted Data')

        # Combinaison des tracés dans une figure Plotly
        fig = go.Figure(data=[trace_train, trace_test, trace_predictions])

        # Mise à jour du layout pour ajouter un titre et des légendes d'axes
        fig.update_layout(
            title='Crimes in Chicago: Actual vs Predicted',
            xaxis_title='Date',
            yaxis_title='Number of Crimes',
            hovermode='x'
        )

        # Affichage de la figure interactive
        fig.show()