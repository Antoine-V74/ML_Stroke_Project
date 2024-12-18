from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def svr_model(X_train, Y_train, X_test, Y_test):
    
    svr = MultiOutputRegressor(SVR(kernel='rbf', C=1.0, epsilon=0.1))
    svr.fit(X_train, Y_train)
    Y_pred = svr.predict(X_test)
    metrics={
        'MAE': mean_absolute_error(Y_test, Y_pred),
        'RMSE': root_mean_squared_error(Y_test, Y_pred),  # Root Mean Squared Error
    }
    return metrics, Y_pred
