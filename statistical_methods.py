import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
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

def filter_chi2(df, target_name, k=8):
    X = df.drop(columns=[target_name]).copy()
    y = _safe_y_int(df[target_name])
    
    try:
        Xm = X - X.min().min()
    except Exception:
        Xm = X.copy()
    
    selector = SelectKBest(chi2, k=min(k, X.shape[1]))
    try:
        selector.fit(Xm.abs(), y)
        mask = selector.get_support()
        selected = X.columns[mask].tolist()
        scores = dict(zip(X.columns, selector.scores_))
    except Exception:
        selected, scores = [], {}
    
    model = LogisticRegression(max_iter=500)
    cv_score = _safe_cv_score(model, X[selected], y) if selected else None
    
    result = {
        'method': 'filter_chi2', 
        'selected_features': selected, 
        'meta': {'scores': _convert_numpy_types(scores)}, 
        'cv_score': cv_score
    }
    
    del X, y, selector
    gc.collect()
    
    return _convert_numpy_types(result)

def mutual_info(df, target_name, k=8):
    X = df.drop(columns=[target_name])
    y = _safe_y_int(df[target_name])
    
    try:
        selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
        selector.fit(X, y)
        mask = selector.get_support()
        selected = X.columns[mask].tolist()
        scores = dict(zip(X.columns, selector.scores_))
    except Exception:
        selected, scores = [], {}
    
    model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
    cv_score = _safe_cv_score(model, X[selected], y) if selected else None
    
    result = {
        'method': 'mutual_info', 
        'selected_features': selected, 
        'meta': {'scores': _convert_numpy_types(scores)}, 
        'cv_score': cv_score
    }
    
    del X, y, selector
    gc.collect()
    
    return _convert_numpy_types(result)

def f_classif_filter(df, target_name, k=8):
    X = df.drop(columns=[target_name])
    y = _safe_y_int(df[target_name])
    
    try:
        selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
        selector.fit(X, y)
        mask = selector.get_support()
        selected = X.columns[mask].tolist()
        scores = dict(zip(X.columns, selector.scores_))
    except Exception:
        selected, scores = [], {}
    
    model = LogisticRegression(max_iter=500)
    cv_score = _safe_cv_score(model, X[selected], y) if selected else None
    
    result = {
        'method': 'f_classif', 
        'selected_features': selected, 
        'meta': {'scores': _convert_numpy_types(scores)}, 
        'cv_score': cv_score
    }
    
    del X, y, selector
    gc.collect()
    
    return _convert_numpy_types(result)

def variance_threshold(df, target_name, threshold=0.01):
    X = df.drop(columns=[target_name])
    y = _safe_y_int(df[target_name])
    
    try:
        selector = VarianceThreshold(threshold=threshold)
        Xs = selector.fit_transform(X)
        mask = selector.get_support()
        selected = X.columns[mask].tolist()
    except Exception:
        selected = []
    
    model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
    cv_score = _safe_cv_score(model, X[selected], y) if selected else None
    
    result = {
        'method': 'variance_threshold', 
        'selected_features': selected, 
        'meta': {'threshold': _convert_numpy_types(threshold)}, 
        'cv_score': cv_score
    }
    
    del X, y, selector
    gc.collect()
    
    return _convert_numpy_types(result)
"""Create By Noor Bakir"""
