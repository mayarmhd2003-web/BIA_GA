
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif, SelectFromModel, VarianceThreshold, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

def _convert_numpy_types(obj):
    """تحويل أنواع NumPy إلى أنواع Python قابلة للتحويل إلى JSON"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        # تحويل السلسلة إلى قائمة مع تحويل القيم
        return obj.astype(float).tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

def _safe_y_int(y):
    """تحويل المتغير الهدف إلى أعداد صحيحة بشكل آمن"""
    try:
        return y.astype(int)
    except Exception:
        return pd.to_numeric(y, errors='coerce').fillna(0).astype(int)

def _safe_cv_score(model, X, y, cv=5):
    """حساب دقة التحقق المتقاطع بشكل آمن مع تحويل النتيجة"""
    try:
        score = np.mean(cross_val_score(model, X, y, cv=cv, scoring='accuracy'))
        return _convert_numpy_types(score)
    except Exception:
        return None

def embedding_rf(df, target_name, top_k=None):
    """اختيار الميزات باستخدام أهمية الميزات في Random Forest"""
    X = df.drop(columns=[target_name])
    y = _safe_y_int(df[target_name])
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    try:
        model.fit(X, y)
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    except Exception:
        importances = pd.Series(0, index=X.columns)

    # تحديد عدد الميزات المختارة
    if top_k is None:
        top_k = max(1, X.shape[1] // 4)
    selected = importances.iloc[:top_k].index.tolist()
    cv_score = _safe_cv_score(model, X[selected], y)
    
    result = {
        'method': 'embedding_rf', 
        'selected_features': selected, 
        'meta': {'importances': _convert_numpy_types(importances.to_dict())}, 
        'cv_score': cv_score
    }
    
    return _convert_numpy_types(result)

def filter_chi2(df, target_name, k=10):
    """اختيار الميزات باستخدام اختبار Chi-square"""
    X = df.drop(columns=[target_name]).copy()
    y = _safe_y_int(df[target_name])
    
    # تحويل القيم لتكون غير سالبة (مطلوب لـ Chi-square)
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
    
    # تقييم النموذج
    model = LogisticRegression(max_iter=1000)
    cv_score = None
    if selected:
        cv_score = _safe_cv_score(model, X[selected], y)
    
    result = {
        'method': 'filter_chi2', 
        'selected_features': selected, 
        'meta': {'scores': _convert_numpy_types(scores)}, 
        'cv_score': cv_score
    }
    
    return _convert_numpy_types(result)

def mutual_info(df, target_name, k=10):
    """اختيار الميزات باستخدام المعلومات المتبادلة"""
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
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_score = _safe_cv_score(model, X[selected], y) if selected else None
    
    result = {
        'method': 'mutual_info', 
        'selected_features': selected, 
        'meta': {'scores': _convert_numpy_types(scores)}, 
        'cv_score': cv_score
    }
    
    return _convert_numpy_types(result)

def f_classif_filter(df, target_name, k=10):
    """اختيار الميزات باستخدام اختبار F (ANOVA)"""
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
    
    model = LogisticRegression(max_iter=1000)
    cv_score = _safe_cv_score(model, X[selected], y) if selected else None
    
    result = {
        'method': 'f_classif', 
        'selected_features': selected, 
        'meta': {'scores': _convert_numpy_types(scores)}, 
        'cv_score': cv_score
    }
    
    return _convert_numpy_types(result)

def l1_logistic(df, target_name, C=0.1):
    """اختيار الميزات باستخدام الانحدار اللوجستي مع عقوبة L1"""
    X = df.drop(columns=[target_name])
    y = _safe_y_int(df[target_name])
    
    print(f"🔍 تشغيل L1 Logistic مع C={C}")
    print(f"📊 شكل البيانات: {X.shape}")
    
    try:
        # تحجيم البيانات
        scaler = MinMaxScaler()
        Xs = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        # استخدام solver مناسب لـ L1
        model = LogisticRegression(
            penalty='l1', 
            solver='liblinear',  # liblinear أفضل لـ L1
            C=C, 
            max_iter=1000,
            random_state=42
        )
        
        selector = SelectFromModel(model, threshold='mean')
        selector.fit(Xs, y)
        mask = selector.get_support()
        selected = X.columns[mask].tolist()
        
        # الحصول على معاملات النموذج
        if hasattr(selector.estimator_, 'coef_'):
            importances = np.abs(selector.estimator_.coef_.ravel())
            importances_dict = dict(zip(X.columns, importances))
        else:
            importances_dict = {}
            
        print(f"✅ L1 Logistic: تم اختيار {len(selected)} ميزة")
        print(f"📋 الميزات المختارة: {selected}")
        
    except Exception as e:
        print(f"❌ خطأ في L1 Logistic: {str(e)}")
        selected, importances_dict = [], {}
    
    # تقييم النموذج
    cv_model = LogisticRegression(max_iter=1000, random_state=42)
    cv_score = _safe_cv_score(cv_model, X[selected], y) if selected else 0.0
    
    # إذا لم يتم اختيار أي ميزات، نستخدم جميع الميزات للتقييم
    if not selected:
        cv_score = _safe_cv_score(cv_model, X, y) or 0.0
        print("⚠️  لم يتم اختيار أي ميزات في L1 Logistic، استخدام جميع الميزات للتقييم")
    
    result = {
        'method': 'l1_logistic', 
        'selected_features': selected, 
        'meta': {'importances': _convert_numpy_types(importances_dict)}, 
        'cv_score': cv_score
    }
    
    print(f"🎯 نتيجة L1 Logistic: {cv_score}")
    return _convert_numpy_types(result)

def rfe_rf(df, target_name, n_features_to_select=None):
    """اختيار الميزات باستخدام الإزالة العودية للميزات مع Random Forest"""
    X = df.drop(columns=[target_name])
    y = _safe_y_int(df[target_name])
    n_total = X.shape[1]
    
    if n_features_to_select is None:
        n_features_to_select = max(1, n_total // 4)
    
    try:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
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
    
    return _convert_numpy_types(result)

def variance_threshold(df, target_name, threshold=0.0):
    """إزالة الميزات ذات التباين المنخفض"""
    X = df.drop(columns=[target_name])
    y = _safe_y_int(df[target_name])
    try:
        selector = VarianceThreshold(threshold=threshold)
        Xs = selector.fit_transform(X)
        mask = selector.get_support()
        selected = X.columns[mask].tolist()
    except Exception:
        selected = []
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_score = _safe_cv_score(model, X[selected], y) if selected else None
    
    result = {
        'method': 'variance_threshold', 
        'selected_features': selected, 
        'meta': {'threshold': _convert_numpy_types(threshold)}, 
        'cv_score': cv_score
    }
    
    return _convert_numpy_types(result)