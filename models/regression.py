from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_predict
import numpy as np

def ridge_multi_out_model(X, Y, cv=5, n_iter=6):
    """
    Ridge Regression for Multi-Output with RandomizedSearchCV.
    """
    ridge = MultiOutputRegressor(Ridge())

    # Define parameter grid
    param_distributions = {
        'estimator__alpha': [0.01, 0.1, 1, 10, 100, 1000]
    }

    random_search = RandomizedSearchCV(ridge, param_distributions, n_iter=n_iter, 
                                       scoring='neg_mean_squared_error', cv=cv, random_state=42)
    random_search.fit(X, Y)
    # Best model and prediction
    best_ridge = random_search.best_estimator_
    Y_pred = cross_val_predict(best_ridge, X, Y, cv=cv)
    
    metrics = {
        'MAE': mean_absolute_error(Y, Y_pred),
        'RMSE': root_mean_squared_error(Y, Y_pred)
    }
    return metrics, Y_pred


# Gradient Boosting for Multi-Output
def gradient_boost_multi_out_model(X, Y, cv=5, n_iter=20):
    """
    Gradient Boosting Multi-Output Regression with RandomizedSearchCV.
    """
    gb = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
    param_distributions = {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'estimator__max_depth': [3, 5, 7],
        'estimator__min_samples_split': [2, 5, 10]
    }

    random_search = RandomizedSearchCV(gb, param_distributions, n_iter=n_iter, 
                                       scoring='neg_mean_squared_error', cv=cv, random_state=42)
    random_search.fit(X, Y)

    # Best model and prediction
    best_gb = random_search.best_estimator_
    Y_pred = cross_val_predict(best_gb, X, Y, cv=cv)
    metrics = {
        'MAE': mean_absolute_error(Y, Y_pred),
        'RMSE': root_mean_squared_error(Y, Y_pred)  # Root Mean Squared Error
    }
    return metrics, Y_pred
