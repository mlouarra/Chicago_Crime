import mlflow

from prophet import Prophet

def train_model(df_train, df_test, params):
    # Optimisation des hyperparamètres

    model = Prophet(**params)
    model.fit(df_train)
    forecast = model.predict(df_test)
    predictions = forecast['yhat']
    return predictions, params, model


