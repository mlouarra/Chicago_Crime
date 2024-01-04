import joblib
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

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
        'Community Area Number':'community_area_number',
        'COMMUNITY AREA NAME':'community_area_name',
        'PERCENT OF HOUSING CROWDED':'pct_housing_crowded',
        'PERCENT HOUSEHOLDS BELOW POVERTY':'pct_households_below_poverty',
        'PERCENT AGED 16+ UNEMPLOYED':'pct_age16_unemployed',
        'PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA':'pct_age25_no_highschool',
        'PERCENT AGED UNDER 18 OR OVER 64': 'pct_not_working_age',
        'per_capita_income':'per_capita_income',
        'HARDSHIP INDEX' : 'hardship_index'}

    def __init__(self, months_pred, data_dir):

        """
        Initialise l'instance de ChicagoCrimePredictor.

        :param months_pred: Nombre de mois pour les prédictions futures.
        :param data_dir: Chemin du répertoire contenant les fichiers de données.
        """
        self.model = None
        self._month_pred = months_pred
        self.data_dir = Path(data_dir)


    def load_df_crimes(self):

        """

        Charge les données sur les crimes depuis un fichier CSV.

        :return: DataFrame contenant les données sur les crimes.
        """

        crimes_file_path = self.data_dir / 'Crimes_Chicago.csv'
        df_crimes = pd.read_csv(crimes_file_path, usecols=['Date', 'Primary Type', 'Community Area'], parse_dates=['Date'])
        df_crimes.rename(columns=self.dicto_rename_crimes, inplace=True)
        return df_crimes

    def load_df_socio(self):
        """

        Charge les données socio-économiques depuis un fichier CSV.

        :return: DataFrame contenant les données socio-économiques.
        """
        socio_file_path = self.data_dir / 'socio_economic_Chicago.csv'
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
        df_socio = self.load_df_socio()
        # Fusion des DataFrames sur 'community_area_number' avec une jointure gauche.
        df = pd.merge(df_crimes, df_socio[['community_area_number', 'community_area_name']],
                      on='community_area_number', how='left')
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
        # Affichage des métriques d'évaluation
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f'R²: {r2}')

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