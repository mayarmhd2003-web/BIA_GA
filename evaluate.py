import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

sns.set_style('whitegrid')

def _convert_numpy_types(obj):
    """تحويل أنواع NumPy إلى أنواع Python قابلة للتحويل إلى JSON"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.astype(float).tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

def _safe_to_int_series(s):
    try:
        return s.astype(int)
    except Exception:
        return pd.to_numeric(s, errors='coerce').fillna(0).astype(int)

def _compute_chi_pvalues_once(X, y, features):
    """
    حساب p-value لتشي-سكوير لكل ميزة في features مرة واحدة فقط.
    يعيد dict {feature: pvalue or None}
    """
    pvals = {}
    for feat in features:
        try:
            # تجزئة إلى 4 فئات متساوية قبل الجداء التبادلي
            cont = pd.crosstab(pd.qcut(X[feat], q=4, duplicates='drop'), y)
            chi2, p, dof, ex = chi2_contingency(cont)
            pvals[feat] = float(p)
        except Exception:
            pvals[feat] = None
    return pvals

def compare_and_stats(df, target_name, methods_results):
    """
    يعيد ملخص مقارن لكل طريقة:
    - fitness_score: final_score أو cv_score أو تقييم بسيط عبر RF
    - n_features: عدد الميزات المختارة
    - selected_features: قائمة الميزات
    - chi_pvalues: p-values لكل ميزة (محسوبة مرة واحدة لكل طريقة)
    """
    X = df.drop(columns=[target_name])
    y = _safe_to_int_series(df[target_name])
    summary = {}

    for res in methods_results:
        method = res.get('method', 'unknown')
        sel = res.get('selected_features', []) or []

        # تحديد score بطريقة مرتبة (final_score ثم cv_score ثم تقييم RF)
        score = None
        if 'final_score' in res:
            try:
                score = float(res['final_score'])
            except Exception:
                score = None
        elif 'cv_score' in res:
            try:
                score = float(res['cv_score'])
            except Exception:
                score = None
        else:
            if sel:
                try:
                    m = RandomForestClassifier(n_estimators=100, random_state=42)
                    m.fit(X[sel], y)
                    preds = m.predict(X[sel])
                    score = float(accuracy_score(y, preds))
                except Exception:
                    score = None
            else:
                score = None

        # حساب chi-square p-values مرة واحدة لكل هذه الطريقة (إذا كانت هناك ميزات محددة)
        chi_pvalues = _compute_chi_pvalues_once(X, y, sel) if sel else {}

        # إضافة الملخص
        summary[method] = {
            'method': method,
            'fitness_score': score,
            'n_features': len(sel),
            'selected_features': sel,
            'chi_pvalues': chi_pvalues
        }

    return _convert_numpy_types(summary)

def plot_results_base64(df, target_name, methods_results):
    """
    يرسم:
    - comparison_bar: شريطي يقارن fitness لكل طريقة
    - ga_history: إذا وُجد history لطريقة 'genetic'، يرسم تطور اللياقة
    يعيد قاموس به الصور base64
    """
    summary = compare_and_stats(df, target_name, methods_results)
    names = []
    scores = []
    for k, v in summary.items():
        names.append(k)
        scores.append(v['fitness_score'] if v['fitness_score'] is not None else 0.0)

    results = {}

    # رسم شريطي للمقارنة
    plt.figure(figsize=(7,4))
    ax = sns.barplot(x=names, y=scores, palette='crest')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Fitness Score')
    ax.set_xlabel('Method')
    plt.xticks(rotation=15)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    plt.close()
    buf.seek(0)
    results['comparison_bar'] = base64.b64encode(buf.read()).decode('utf-8')

    # رسم تطور اللياقة للـ GA إذا توفر history (نأخذ أول نتيجة method=='genetic')
    for res in methods_results:
        if res.get('method') == 'genetic' and res.get('history'):
            history = res['history']
            try:
                gens = [h['generation'] for h in history]
                fits = [h['best_fitness'] for h in history]
                plt.figure(figsize=(7,3))
                plt.plot(gens, fits, marker='o', linestyle='-', color='#2b8cbe')
                plt.fill_between(gens, fits, alpha=0.1, color='#2b8cbe')
                plt.title('Fitness Evolution Over Generations')
                plt.xlabel('Generation')
                plt.ylabel('Fitness Value')
                ymin = min(fits) - 0.05 if fits else 0
                ymax = max(fits) + 0.05 if fits else 1
                plt.ylim(max(0, ymin), min(1.05, ymax))
                plt.tight_layout()
                buf2 = io.BytesIO()
                plt.savefig(buf2, format='png', dpi=120)
                plt.close()
                buf2.seek(0)
                results['ga_history'] = base64.b64encode(buf2.read()).decode('utf-8')
            except Exception:
                results['ga_history'] = None
            break

    return results