# -*- coding: utf-8 -*-
import pandas as pd
import json
import yaml


pd.options.mode.chained_assignment = None

class LoadDataframe:

    """

    """

    def __init__(self, config, start_year, end_year):
        """

        :param config:
        """
        self._config = config
        self._start_year = pd.to_datetime(str(start_year))
        self._end_year = pd.to_datetime(str(end_year))


    def path_crime(self):
        """

        :return:
        """
        return self._config['connect']['PathCrimes']

    def path_socio(self):
        """

        :return:
        """
        return self._config['connect']['PathSocioEco']

    def path_columns(self):
        """

        :return:
        """
        return self._config['connect']['Pathcolumns']

    def path_temperature(self):
        """

        :return:
        """

        return self._config['connect']['PathTemperature']

    def path_sky(self):
        """

        :return:
        """
        return self._config['connect']['PathSky']

    def df_crime(self):
        """

        :return:
        """
        column_name = json.loads(open(self.path_columns()).read())
        df_crime = pd.read_csv(self.path_crime(), parse_dates=['Date'])
        df_crime.rename(columns=column_name['DataCrimes'], inplace=True)

        if self._config["List_of_crimes_prediction"]["with_merge"]:
            df_crime.replace({'primary_type': self._config["List_of_crimes_prediction"]["to_merge"]}, inplace=True)
            return df_crime

        else:
            return df_crime

    def df_socio(self):
        """

        :return:
        """
        from sklearn.preprocessing import MinMaxScaler
        min_max_scaler = MinMaxScaler()
        column_name = json.loads(open(self.path_columns()).read())
        df_socio = pd.read_csv(self.path_socio())
        df_socio.rename(columns=column_name['SocioEco'], inplace=True)
        column_names_to_normalize = ['pct_housing_crowded','pct_households_below_poverty',  'pct_age16_unemployed' , 'pct_age25_no_highschool', 'pct_not_working_age','per_capita_income',
                'hardship_index']
        x = df_socio[column_names_to_normalize].values
        x_scaled = min_max_scaler.fit_transform(x)
        df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index=df_socio.index)
        df_socio[column_names_to_normalize] = df_temp
        del df_temp
        return df_socio

    def df_crime_socio(self):
        column_name = json.loads(open(self.path_columns()).read())
        df_crime = pd.read_csv(self.path_crime(), sep=';')
        df_crime.rename(columns=column_name['DataCrimes'], inplace=True)
        df_socio = self.df_socio()
        df_merged = pd.merge(df_crime, df_socio, on='community_area_number', how='left')

        return df_merged

    def df_temperature(self):
        """

        :return:
        """

        pd.options.mode.chained_assignment = None
        df = pd.read_csv(self.path_temperature(), parse_dates=['datetime'])
        df = df[['datetime', 'Chicago']]
        df.rename(columns={'Chicago': 'Temperature'}, inplace=True)
        df = df[(self._start_year <= df['datetime']) & (df['datetime'] < self._end_year)]
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hours'] = df['datetime'].dt.hour
        df.drop(columns='datetime', inplace=True, axis=1)

        return df

    def df_sky(self):
        """

        :return:
        """
        df = pd.read_csv(self.path_sky(), parse_dates=['datetime'])
        df = df[['Chicago', 'datetime']]
        df.dropna(inplace=True)
        df = pd.get_dummies(df, columns=['Chicago'])
        df = df[(self._start_year <= df['datetime']) & (df['datetime'] < self._end_year)]
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hours'] = df['datetime'].dt.hour
        df.drop(columns='datetime', inplace=True, axis=1)
        return df

    def df_merged(self):
        """

        :return:
        """
        column_name = json.loads(open(self.path_columns()).read())
        df_crime = pd.read_csv(self.path_crime(), sep=';')
        df_crime.rename(columns=column_name['DataCrimes'], inplace=True)
        df_socio = self.df_socio()
        df_merged = pd.merge(df_crime, df_socio, on='community_area_number', how='left')
        return df_merged

    def df_crime_socio(self):

        """

        :param year:
        :return:
        """

        df_crime = self.df_crime()
        df_socio = self.df_socio()
        df_crime = df_crime[(self._start_year <= df_crime['date']) & (df_crime['date'] < self._end_year)]
        df_src = pd.merge(df_crime, df_socio, on='community_area_number', how='left')
        del df_crime
        del df_socio
        return df_src

    def df_nb_crimes(self):

        """

        :param year:
        :return:
        """

        df_S = self.df_socio()
        df_C = self.df_crime()
        df_C = df_C[df_C.primary_type.isin(self._config["NameCrime"])]
        list_name_crimes = list(df_C['primary_type'].unique())
        df_year = df_C[(self._start_year <= df_C['date']) & (df_C['date'] < self._end_year)]
        df_year['month'] = pd.DatetimeIndex(df_year['date']).month
        df_year['year'] = pd.DatetimeIndex(df_year['date']).year
        df_year_grouped = df_year.groupby(['community_area_number', 'month', 'year', 'primary_type'],
                                          as_index=False).agg({'id': 'count'})
        df_year_grouped.rename(columns={'id': 'nb_crimes'}, inplace=True)
        df_merged = pd.merge(df_year_grouped, df_S, on='community_area_number', how='inner')
        del df_C
        del df_S
        df_merged.dropna(inplace=True)
        df_merged.drop(['year', 'community_area_number'], axis=1, inplace=True)
        df_merged_ = pd.get_dummies(df_merged, columns=['primary_type', 'community_area_name', 'month'])
        del df_year_grouped
        del df_merged
        for col in list_name_crimes:
            if "primary_type_" + col not in list(df_merged_.columns):
                df_merged_["primary_type_" + col] = -1
        return df_merged_

if __name__ == "__main__":

    path_config = '../../config/config.yml'
    with (open(path_config, 'r') as fichier):
        # Utilisez la fonction load() pour charger le contenu du fichier YAML
        config = yaml.load(fichier, Loader=yaml.FullLoader)
    obj_df = LoadDataframe(config, 2020, 2023)
    df_nb_crimes = obj_df.df_nb_crimes()

    print("df_nb_crimes", df_nb_crimes)


