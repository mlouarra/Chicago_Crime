from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from prophet import Prophet
from sklearn.metrics import mean_squared_error

def optimize_hyperparameters(df_train, df_test):
    # Définition de la fonction objectif
    @use_named_args(dimensions=[Real(1e-2, 1e2, name='seasonality_prior_scale'),
                                 Real(1e-4, 1e-1, name='changepoint_prior_scale')])
    def objective(seasonality_prior_scale, changepoint_prior_scale):
        # Création du modèle avec les hyperparamètres spécifiés
        model = Prophet(seasonality_prior_scale=seasonality_prior_scale,
                        changepoint_prior_scale=changepoint_prior_scale)
        # Entraînement du modèle
        model.fit(df_train)
        # Prédictions sur les données de test
        forecast = model.predict(df_test)
        # Calcul de l'erreur quadratique moyenne
        rmse = mean_squared_error(df_test['y'], forecast['yhat'], squared=False)
        # Retour de l'erreur quadratique moyenne (RMSE)
        return rmse

    # Définition de l'espace de recherche des hyperparamètres
    space = [Real(1e-2, 1e2, name='seasonality_prior_scale'),
             Real(1e-4, 1e-1, name='changepoint_prior_scale')]

    # Optimisation des hyperparamètres
    result = gp_minimize(objective, space, n_calls=50, random_state=42)

    # Récupération des meilleurs hyperparamètres
    best_seasonality_prior_scale = result.x[0]
    best_changepoint_prior_scale = result.x[1]

    # Stockage des meilleurs hyperparamètres dans un dictionnaire
    best_params = {
        "seasonality_prior_scale": best_seasonality_prior_scale,
        "changepoint_prior_scale": best_changepoint_prior_scale
    }

    # Affichage des meilleurs hyperparamètres trouvés
    print("Best hyperparameters found:")
    print("seasonality_prior_scale =", best_seasonality_prior_scale)
    print("changepoint_prior_scale =", best_changepoint_prior_scale)

    # Retourne les meilleurs hyperparamètres
    return best_params
