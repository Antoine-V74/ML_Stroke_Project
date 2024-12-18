from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

def logistic_regression_model(X, y, cv=5, n_iter=6):
    """
    Logistic Regression with RandomizedSearchCV for hyperparameter tuning.
    """
    lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=5000)
    
    # Define parameter grid
    param_distributions = {
        'C': np.logspace(-4, 4, 10),
        'penalty': ['l1','l2'],
        'solver': ['saga']
    }

    random_search = RandomizedSearchCV(lr, param_distributions, n_iter=n_iter, 
                                       scoring='f1', cv=cv, random_state=42)
    random_search.fit(X, y)

    # Best model and prediction
    best_lr = random_search.best_estimator_
    y_pred = cross_val_predict(best_lr, X, y, cv=cv)

    # Compute metrics
    metrics = {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, average='weighted'),
        'Recall': recall_score(y, y_pred, average='weighted'),
        'F1-Score': f1_score(y, y_pred, average='weighted'),
    }
    return metrics, y_pred
