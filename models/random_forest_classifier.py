from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def random_forest_model(X, y, cv=5, n_iter=10):
    """
    Random Forest Classifier with RandomizedSearchCV.
    """
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)

    # Define parameter grid
    param_distributions = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    random_search = RandomizedSearchCV(rf, param_distributions, n_iter=n_iter, 
                                       scoring='f1', cv=cv, random_state=42)
    random_search.fit(X, y)

    # Best model and prediction
    best_rf = random_search.best_estimator_
    y_pred = cross_val_predict(best_rf, X, y, cv=cv)

    # Compute metrics
    metrics = {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, average='weighted'),
        'Recall': recall_score(y, y_pred, average='weighted'),
        'F1-Score': f1_score(y, y_pred, average='weighted'),
    }
    return metrics, y_pred
