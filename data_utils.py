import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

def generate_random_dataset(n_rows=200, n_cols=30, n_informative=10, random_state=None):
    X, y = make_classification(n_samples=n_rows, n_features=n_cols,
                               n_informative=n_informative, n_redundant=0,
                               n_repeated=0, n_classes=2, random_state=random_state,
                               shuffle=True)
    cols = [f'feature_{i+1}' for i in range(n_cols)]
    df = pd.DataFrame(X, columns=cols)
    df['target'] = y
    return df, 'target'

def preprocess_dataset(df, target_name=None):
    df = df.copy()
    if target_name is None:
        if 'target' in df.columns:
            target_name = 'target'
        else:
            target_name = df.columns[-1]
    df = df.dropna(axis=1, how='all')
    X = df.drop(columns=[target_name])
    y = df[target_name]
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    for c in X_imputed.columns:
        try:
            X_imputed[c] = pd.to_numeric(X_imputed[c])
        except Exception:
            X_imputed[c] = X_imputed[c].astype('category').cat.codes
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)
    # التأكد أن y متوافق نوعياً مع التصنيف
    try:
        y = pd.to_numeric(y)
    except Exception:
        y = y.astype('category').cat.codes
    df_clean = pd.concat([X_scaled, pd.Series(y, name=target_name)], axis=1)
    return df_clean, target_name
