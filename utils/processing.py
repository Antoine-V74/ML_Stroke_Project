import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def Processing_std(df):
    
    scaler=StandardScaler()
    df=scaler.fit_transform(df)
    
    return df

def Processing_Min_Max(df):
    
    scaler=MinMaxScaler()
    df=scaler.fit_transform(df)
    return df

def Processing_PCA(df, n_components=0.8):
    
    scaler=StandardScaler()
    pca = PCA(n_components=n_components)
    scaler.fit_transform(df)
    Y_reduced = pca.fit_transform(df)
    explained_variance = pca.explained_variance_ratio_

    return pd.DataFrame(Y_reduced), explained_variance

def Processing_collinear(df,threshold=0.6):
    
    df=remove_collinear_features(df, threshold)
    list_col=df.columns.tolist()
    df.drop(columns=[x for x in list_col if (x.startswith("Ardila") or x.startswith("Token") or x.startswith("Door"))])
    scaler=StandardScaler()
    df_corr=scaler.fit_transform(df)
    
    return df_corr

def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model 
        to generalize and improves the interpretability of the model.

    Inputs: 
        x: features dataframe
        threshold: features with correlations greater than this value are removed

    Output: 
        dataframe that contains only the non-highly-collinear features
    '''

    # Calculate the correlation matrix
    df=x.copy()
    corr_matrix = df.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                #print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    df = df.drop(columns=drops)
    #print('Removed Columns {}'.format(drops))
    return df

def get_common_patients(path_T1, path_T3):
    """
    Identifies common patients in T1 and T3 datasets.
    Args:
        path_T1 (str): Directory path for T1 connectomes.
        path_T3 (str): Directory path for T3 connectomes.
    Returns:
        set: Set of patient IDs common to both T1 and T3.
    """
    patients_T1 = set([f.split("_")[0] for f in os.listdir(path_T1) if f.endswith(".csv")])
    patients_T3 = set([f.split("_")[0] for f in os.listdir(path_T3) if f.endswith(".csv")])
    return patients_T1.intersection(patients_T3)

def Processing_connectomes(path_dir, common_patients):
    """
    Processes connectomes and filters for common patients.
    Args:
        path_dir (str): Directory path for connectomes.
        common_patients (set): Set of common patient IDs.
    Returns:
        dict: Dictionary with patient IDs as keys and flattened connectomes as values.
    """
    patient_connectomes = {}
    for f in os.listdir(path_dir):
        if f.endswith(".csv"):
            patient_id = f.split("_")[0]
            if patient_id in common_patients:
                file_path = os.path.join(path_dir, f)
                connectome = pd.read_csv(file_path)
                features_list = connectome.columns[1:-1]
                connectome_values = connectome[features_list].values.flatten()
                patient_connectomes[patient_id] = connectome_values
    return patient_connectomes

def Processing_rf_features(X, Y, threshold=0.02):
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, Y.mean(axis=1))  
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
    selected_features = feature_importances[feature_importances > threshold].index
    f_i=feature_importances[selected_features].sort_values()

    return X[selected_features], feature_importances.sort_values(ascending=False)


def Processing_PLS(X, Y, n_components=5):
    """
    Reduce dimensionality of targets Y using Partial Least Squares (PLS).
    Args:
        X: Feature matrix (n_samples, n_features)
        Y: Target matrix (n_samples, n_targets)
        n_components: Number of components for PLS.
    Returns:
        Y_reduced: Reduced targets
        pls_model: Fitted PLS model (for reconstruction if needed)
    """
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, Y)
    Y_reduced = pls.transform(Y)  # Reduced target dimensions
    return pd.DataFrame(Y_reduced), pls


def Processing_imbalance(X, y):
    """Handle imbalanced data with SMOTE."""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


