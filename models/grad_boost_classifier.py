from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def gradient_boosting_model(X, y, cv=5, n_iter=10):
    """
    Gradient Boosting Classifier with RandomizedSearchCV.
    """
    gb = GradientBoostingClassifier(random_state=42)

    # Define parameter grid
    param_distributions = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    }

    random_search = RandomizedSearchCV(gb, param_distributions, n_iter=n_iter, 
                                       scoring='f1', cv=cv, random_state=42)
    random_search.fit(X, y)

    # Best model and prediction
    best_gb = random_search.best_estimator_
    y_pred = cross_val_predict(best_gb, X, y, cv=cv)

    # Compute metrics
    metrics = {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, average='weighted'),
        'Recall': recall_score(y, y_pred, average='weighted'),
        'F1-Score': f1_score(y, y_pred, average='weighted'),
    }
    return metrics, y_pred
