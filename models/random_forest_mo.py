from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_predict

def random_forest_multi_out_model(X, Y, cv=5, n_iter=20):
    """
    Random Forest Multi-Output Regression with RandomizedSearchCV.
    """
    rf = MultiOutputRegressor(RandomForestRegressor(random_state=42))

    # Define parameter grid for RandomizedSearch
    param_distributions = {
        'estimator__n_estimators': [50, 100, 200, 500],
        'estimator__max_depth': [None, 10, 20, 30],
        'estimator__min_samples_split': [2, 5, 10],
        'estimator__min_samples_leaf': [1, 2, 4]
    }
    
    random_search = RandomizedSearchCV(rf, param_distributions, n_iter=n_iter, 
                                       scoring='neg_mean_squared_error', cv=cv, random_state=42)
    random_search.fit(X, Y)

    # Best model and prediction
    best_rf = random_search.best_estimator_
    Y_pred = cross_val_predict(best_rf, X, Y, cv=cv)
    
    metrics = {
        'MAE': mean_absolute_error(Y, Y_pred),
        'RMSE': root_mean_squared_error(Y, Y_pred)
    }
    return metrics, Y_pred