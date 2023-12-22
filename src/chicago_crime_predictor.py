import joblib
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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

    def __init__(self, months_pred):
        self.model = None
        self._month_pred = months_pred


    def load_df_crimes(self):

        df_crimes = pd.read_csv("../data/raw/Crimes_Chicago.csv", usecols=['Date', 'Primary Type', 'Community Area'], parse_dates=['Date'])
        df_crimes.rename(columns=self.dicto_rename_crimes, inplace=True)
        return df_crimes

    def load_df_socio(self):

        df_socio = pd.read_csv("../data/raw/socio_economic_Chicago.csv")
        df_socio.rename(columns=self.dicto_rename_socio, inplace=True)
        return df_socio

    def return_data(self, type_incident, community_area=None):
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

        # Créer et entraîner le modèle Prophet
        self.model = Prophet(yearly_seasonality=True,
                        # weekly_seasonality=True,
                        daily_seasonality=False,
                        seasonality_prior_scale=10,  # Augmenter pour une saisonnalité plus flexible
                        changepoint_prior_scale=0.05  # Diminuer pour des tendances moins flexibles)
                        )
        self.model.fit(data_train)


    def model_predict(self):
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné. Utilisez la méthode 'model_train' d'abord.")

        # Prédire les valeurs pour les dates futures
        future = self.model.make_future_dataframe(periods=self._month_pred, freq='M')
        forecast = self.model.predict(future)
        split_date = forecast['ds'].max() - pd.DateOffset(months=self._month_pred)
        predictions = forecast[['ds', 'yhat']].loc[forecast['ds'] > split_date]
        return forecast, predictions

    def model_save(self, filename):
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné. Utilisez la méthode 'model_train' d'abord.")

        # Enregistrer le modèle dans un fichier
        joblib.dump(self.model, filename)
        print(f"Modèle enregistré sous {filename}")

    def model_evaluation(self, test, predictions):

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


# Exemple d'utilisation :
if __name__ == "__main__":
    obj_predict = ChicagoCrimePredictor(12)
    data_ml = obj_predict.return_data("ASSAULT", 15, 'Austin')
    # obj_predict.model_train(data_ml)
    # obj_predict.model_save("../models/model_theft")
    
    print(data_ml)
