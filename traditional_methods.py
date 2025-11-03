import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import gc

def _convert_numpy_types(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.astype(float, errors='ignore').tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

def _safe_y_int(y):
    try:
        return y.astype(int)
    except Exception:
        return pd.to_numeric(y, errors='coerce').fillna(0).astype(int)

def _safe_cv_score(model, X, y, cv=3):
    try:
        score = np.mean(cross_val_score(model, X, y, cv=cv, scoring='accuracy'))
        return _convert_numpy_types(score)
    except Exception:
        return None

def embedding_rf(df, target_name, top_k=None):
    X = df.drop(columns=[target_name])
    y = _safe_y_int(df[target_name])
    
    model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
    try:
        model.fit(X, y)
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    except Exception:
        importances = pd.Series(0, index=X.columns)

    if top_k is None:
        top_k = max(1, X.shape[1] // 3)
    
    selected = importances.iloc[:top_k].index.tolist()
    cv_score = _safe_cv_score(model, X[selected], y)
    
    result = {
        'method': 'embedding_rf', 
        'selected_features': selected, 
        'meta': {'importances': _convert_numpy_types(importances.to_dict())}, 
        'cv_score': cv_score
    }
    
    del X, y, model, importances
    gc.collect()
    
    return _convert_numpy_types(result)

def l1_logistic(df, target_name, C=0.1):
    X = df.drop(columns=[target_name])
    y = _safe_y_int(df[target_name])
    
    selected = []
    importances_dict = {}
    
    try:
        scaler = MinMaxScaler()
        Xs = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        model = LogisticRegression(
            penalty='l1', 
            solver='liblinear',
            C=C, 
            max_iter=500,
            random_state=42
        )
        
        selector = SelectFromModel(model, threshold='mean')
        selector.fit(Xs, y)
        mask = selector.get_support()
        selected = X.columns[mask].tolist()
        
        if hasattr(selector.estimator_, 'coef_'):
            importances = np.abs(selector.estimator_.coef_.ravel())
            importances_dict = dict(zip(X.columns, importances))
    except Exception:
        selected, importances_dict = [], {}
    
    cv_model = LogisticRegression(max_iter=500, random_state=42)
    cv_score = _safe_cv_score(cv_model, X[selected], y) if selected else 0.0
    
    if not selected:
        cv_score = _safe_cv_score(cv_model, X, y) or 0.0
    
    result = {
        'method': 'l1_logistic', 
        'selected_features': selected, 
        'meta': {'importances': _convert_numpy_types(importances_dict)}, 
        'cv_score': cv_score
    }
    
    del X, y, scaler, model, selector
    gc.collect()
    
    return _convert_numpy_types(result)

def rfe_rf(df, target_name, n_features_to_select=None):
    X = df.drop(columns=[target_name])
    y = _safe_y_int(df[target_name])
    n_total = X.shape[1]
    
    if n_features_to_select is None:
        n_features_to_select = max(1, n_total // 3)
    
    try:
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1)
        selector.fit(X, y)
        mask = selector.get_support()
        selected = X.columns[mask].tolist()
        ranking = dict(zip(X.columns, selector.ranking_))
    except Exception:
        selected, ranking = [], {}
    
    cv_score = _safe_cv_score(estimator, X[selected], y) if selected else None
    
    result = {
        'method': 'rfe_rf', 
        'selected_features': selected, 
        'meta': {'ranking': _convert_numpy_types(ranking)}, 
        'cv_score': cv_score
    }
    
    del X, y, estimator, selector
    gc.collect()
    
    return _convert_numpy_types(result)
