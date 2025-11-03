import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import gc

sns.set_style('whitegrid')

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

def _safe_to_int_series(s):
    try:
        return s.astype(int)
    except Exception:
        return pd.to_numeric(s, errors='coerce').fillna(0).astype(int)

def _compute_chi_pvalues_once(X, y, features):
    pvals = {}
    for feat in features:
        try:
            cont = pd.crosstab(pd.qcut(X[feat], q=4, duplicates='drop'), y)
            chi2, p, dof, ex = chi2_contingency(cont)
            pvals[feat] = float(p)
        except Exception:
            pvals[feat] = None
    return pvals

def compare_and_stats(df, target_name, methods_results):
    X = df.drop(columns=[target_name])
    y = _safe_to_int_series(df[target_name])
    summary = {}

    for res in methods_results:
        method = res.get('method', 'unknown')
        sel = res.get('selected_features', []) or []

        score = None
        if 'final_score' in res:
            try:
                score = float(res['final_score'])
            except Exception:
                pass
        elif 'cv_score' in res:
            try:
                score = float(res['cv_score'])
            except Exception:
                pass
        
        if score is None and sel:
            try:
                model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
                model.fit(X[sel], y)
                preds = model.predict(X[sel])
                score = float(accuracy_score(y, preds))
                del model, preds
            except Exception:
                score = None

        chi_pvalues = _compute_chi_pvalues_once(X, y, sel) if sel else {}

        summary[method] = {
            'method': method,
            'fitness_score': score,
            'n_features': len(sel),
            'selected_features': sel,
            'chi_pvalues': chi_pvalues
        }

    del X, y
    gc.collect()
    
    return _convert_numpy_types(summary)

def plot_results_base64(df, target_name, methods_results):
    summary = compare_and_stats(df, target_name, methods_results)
    names = []
    scores = []
    
    for k, v in summary.items():
        names.append(k)
        scores.append(v['fitness_score'] if v['fitness_score'] is not None else 0.0)

    results = {}

    plt.figure(figsize=(7, 4))
    ax = sns.barplot(x=names, y=scores, palette='crest')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Fitness Score')
    ax.set_xlabel('Method')
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    results['comparison_bar'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    for res in methods_results:
        if res.get('method') == 'genetic' and res.get('history'):
            history = res['history']
            try:
                gens = [h['generation'] for h in history]
                fits = [h['best_fitness'] for h in history]
                
                plt.figure(figsize=(7, 3))
                plt.plot(gens, fits, marker='o', linestyle='-', color='#2b8cbe', markersize=3)
                plt.title('Fitness Evolution Over Generations')
                plt.xlabel('Generation')
                plt.ylabel('Fitness Value')
                plt.tight_layout()
                
                buf2 = io.BytesIO()
                plt.savefig(buf2, format='png', dpi=100)
                buf2.seek(0)
                results['ga_history'] = base64.b64encode(buf2.getvalue()).decode('utf-8')
                buf2.close()
                plt.close()
            except Exception:
                results['ga_history'] = None
            break

    gc.collect()
    return results
