import pandas as pd
from sklearn.model_selection import LeaveOneOut

def leave_one_out_validation(X, Y, model_func):
    
    loo = LeaveOneOut()
    metrics_list = []
    for train_index, test_index in loo.split(X):
        # Train-test split for each fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        # Train model and get predictions
        metrics, _ = model_func(X_train, Y_train, X_test, Y_test)
        metrics_list.append(metrics)
    # Aggregate metrics across folds
    avg_metrics = pd.DataFrame(metrics_list).mean().to_dict()
    
    return avg_metrics
