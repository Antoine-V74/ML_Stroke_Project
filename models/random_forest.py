from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def random_forest_model(X_train, Y_train, X_test, Y_test):
    
    regr_rf = RandomForestRegressor(n_estimators=100,  random_state=2)
    regr_rf.fit(X_train, Y_train)
    Y_pred = regr_rf.predict(X_test)
    metrics = {
        'MAE': mean_absolute_error(Y_test, Y_pred),
        'RMSE': root_mean_squared_error(Y_test, Y_pred),  # Root Mean Squared Error
    }
    return metrics, Y_test