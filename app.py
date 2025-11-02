from flask import Flask, render_template, request, jsonify
import pandas as pd
import io
import traceback
import numpy as np
import json
from data_utils import generate_random_dataset, preprocess_dataset
from ga_module import run_genetic_algorithm
import traditional_methods as tm
from evaluate import compare_and_stats, plot_results_base64

app = Flask(__name__)

# معالج JSON مخصص للتعامل مع أنواع NumPy
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        """تحويل أنواع NumPy إلى أنواع Python قابلة للتحويل إلى JSON"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder

# ذاكرة مؤقتة لتخزين النتائج
_cache = {
    'last_ga': None,
    'last_ga_params': {},
    'traditional_methods': {}
}

@app.route('/')
def index():
    """الصفحة الرئيسية للتطبيق"""
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def api_generate():
    """توليد بيانات عشوائية جديدة"""
    try:
        print("🔍 بدء توليد البيانات...")
        payload = request.get_json(force=True) or {}
        print(f"📦 البيانات المستلمة: {payload}")
        
        n_features = int(payload.get('nFeatures', 20))
        n_samples = int(payload.get('nSamples', 200))
        n_informative = int(payload.get('nInformative', max(1, n_features // 4)))
        
        print(f"🔢 المعاملات: n_features={n_features}, n_samples={n_samples}, n_informative={n_informative}")
        
        # التحقق من صحة المدخلات
        if n_features <= 0 or n_samples <= 0 or n_informative <= 0:
            return jsonify({'error': 'القيم يجب أن تكون أكبر من الصفر'}), 400
        
        if n_informative > n_features:
            return jsonify({'error': 'عدد الميزات المعلوماتية يجب أن يكون أقل من أو يساوي إجمالي الميزات'}), 400
        
        df, target = generate_random_dataset(
            n_rows=n_samples, 
            n_cols=n_features, 
            n_informative=n_informative
        )
        
        print(f"✅ تم توليد البيانات بنجاح. الشكل: {df.shape}")
        
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_data = csv_buf.getvalue()
        
        return jsonify({
            'csv': csv_data, 
            'target': target,
            'message': f'تم توليد {n_samples} عينة بـ {n_features} ميزة ({n_informative} معلوماتية)'
        })
    except Exception as e:
        print(f"❌ خطأ في توليد البيانات: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'خطأ في توليد البيانات: {str(e)}'}), 500

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """رفع ملف بيانات من المستخدم"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'لم يتم اختيار ملف'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'لم يتم اختيار ملف'}), 400
        
        name = file.filename.lower()
        try:
            if name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
        except Exception as e:
            return jsonify({'error': 'فشل في قراءة الملف: ' + str(e)}), 400
        
        df_clean, target = preprocess_dataset(df)
        csv_buf = io.StringIO()
        df_clean.to_csv(csv_buf, index=False)
        return jsonify({'csv': csv_buf.getvalue(), 'target': target})
    except Exception as e:
        print(f"❌ خطأ في رفع الملف: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'خطأ في رفع الملف: ' + str(e)}), 500

@app.route('/api/fetch', methods=['POST'])
def api_fetch():
    """جلب بيانات من رابط خارجي"""
    try:
        url = (request.json or {}).get('url')
        if not url:
            return jsonify({'error': 'لم يتم تقديم رابط'}), 400
        
        try:
            df = pd.read_csv(url)
        except Exception:
            try:
                df = pd.read_excel(url)
            except Exception as e:
                return jsonify({'error': 'فشل في جلب البيانات من الرابط: ' + str(e)}), 400
        
        df_clean, target = preprocess_dataset(df)
        csv_buf = io.StringIO()
        df_clean.to_csv(csv_buf, index=False)
        return jsonify({'csv': csv_buf.getvalue(), 'target': target})
    except Exception as e:
        print(f"❌ خطأ في جلب البيانات: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'خطأ في جلب البيانات: ' + str(e)}), 500

def _convert_to_serializable(obj):
    """تحويل الكائنات غير القابلة للتحويل إلى JSON إلى أنواع قابلة للتحويل"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.astype(float).tolist()  # تحويل إلى float أولاً
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

def _read_df_from_payload(payload):
    """قراءة البيانات من payload الطلب"""
    raw = payload.get('df')
    if raw is None:
        raise ValueError('لا توجد بيانات في الطلب')
    
    print(f"🔍 قراءة البيانات من الـ payload. الطول: {len(raw)}")
    
    try:
        # محاولة قراءة كـ CSV (هذا ما نريده)
        df = pd.read_csv(io.StringIO(raw))
        print(f"✅ تم تحميل البيانات كـ CSV بنجاح. الشكل: {df.shape}")
    except Exception as e:
        print(f"❌ فشل في تحميل البيانات كـ CSV: {str(e)}")
        # محاولة قراءة كـ JSON كخيار احتياطي
        try:
            df = pd.read_json(io.StringIO(raw), orient='split')
            print(f"✅ تم تحميل البيانات كـ JSON بنجاح. الشكل: {df.shape}")
        except Exception as json_error:
            print(f"❌ فشل في تحميل البيانات كـ JSON: {str(json_error)}")
            raise ValueError(f'فشل في تحميل البيانات: {str(e)}')
    
    target = payload.get('target')
    if target is None or target not in df.columns:
        target = df.columns[-1]
    
    print(f"✅ الهدف المحدد: {target}")
    return df, target

@app.route('/api/traditional/run', methods=['POST'])
def api_traditional_run():
    """تشغيل طريقة تقليدية لاختيار الميزات"""
    try:
        payload = request.json or {}
        print(f"🔍 تشغيل طريقة تقليدية: {payload.get('method')}")
        
        df, target = _read_df_from_payload(payload)
        method_name = payload.get('method')
        if not method_name:
            return jsonify({'error': 'لم يتم تحديد الطريقة'}), 400

        cache_key = f"{method_name}_{hash(str(df.values.tobytes()) + target)}"
        
        if cache_key in _cache['traditional_methods']:
            cached_result = _cache['traditional_methods'][cache_key]
            print(f"📦 استخدام النتيجة المخزنة للطريقة: {method_name}")
            return jsonify({**cached_result, 'cached': True})

        method_map = {
            'embedding_rf': tm.embedding_rf,
            'filter_chi2': tm.filter_chi2,
            'mutual_info': tm.mutual_info,
            'f_classif': tm.f_classif_filter,
            'l1_logistic': tm.l1_logistic,
            'rfe_rf': tm.rfe_rf,
            'variance_threshold': tm.variance_threshold
        }

        func = method_map.get(method_name)
        if not func:
            return jsonify({'error': f'طريقة غير معروفة: {method_name}'}), 400

        params = payload.get('params', {}) or {}
        print(f"🔧 تشغيل الطريقة: {method_name} مع المعاملات: {params}")
        
        res = func(df, target, **params) if params else func(df, target)
        if 'method' not in res:
            res['method'] = method_name
        
        # تحويل النتائج إلى أنواع قابلة للتحويل إلى JSON
        res = _convert_to_serializable(res)
        
        _cache['traditional_methods'][cache_key] = res
        print(f"✅ تم تشغيل الطريقة بنجاح: {method_name}")
        
        return jsonify({**res, 'cached': False})
    except Exception as e:
        print(f"❌ خطأ في الطريقة التقليدية: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'خطأ في تشغيل الطريقة التقليدية: ' + str(e)}), 500

@app.route('/api/ga', methods=['POST'])
def api_ga():
    """تشغيل الخوارزمية الجينية"""
    try:
        payload = request.json or {}
        print("🔍 بدء تشغيل الخوارزمية الجينية...")
        
        df, target = _read_df_from_payload(payload)
        
        pop_size = int(payload.get('pop_size', 40))
        generations = int(payload.get('generations', 30))
        crossover_rate = float(payload.get('crossover_rate', 0.8))
        mutation_rate = float(payload.get('mutation_rate', 0.02))
        use_cache = bool(payload.get('use_cache', True))

        params = {
            'pop_size': pop_size, 
            'generations': generations,
            'crossover_rate': crossover_rate, 
            'mutation_rate': mutation_rate
        }

        print(f"🔧 معاملات GA: {params}")

        if use_cache and _cache.get('last_ga') is not None and _cache.get('last_ga_params') == params:
            cached = _cache['last_ga']
            print("📦 استخدام نتيجة GA المخزنة")
            return jsonify({**cached, 'cached': True})

        print("🔄 تشغيل الخوارزمية الجينية...")
        res = run_genetic_algorithm(
            df, target, 
            pop_size=pop_size, 
            generations=generations,
            crossover_rate=crossover_rate, 
            mutation_rate=mutation_rate
        )
        
        # تحويل النتائج إلى أنواع قابلة للتحويل إلى JSON
        res = _convert_to_serializable(res)
        
        _cache['last_ga'] = res
        _cache['last_ga_params'] = params
        
        print("✅ تم تشغيل الخوارزمية الجينية بنجاح")
        return jsonify({**res, 'cached': False})
    except Exception as e:
        print(f"❌ خطأ في GA: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'خطأ في الخوارزمية الجينية: ' + str(e)}), 500

@app.route('/api/run_all_traditional', methods=['POST'])
def api_run_all_traditional():
    """تشغيل جميع الطرق التقليدية دفعة واحدة"""
    try:
        payload = request.json or {}
        print("🔍 تشغيل جميع الطرق التقليدية...")
        
        df, target = _read_df_from_payload(payload)
        
        traditional_methods = ['embedding_rf', 'l1_logistic', 'rfe_rf']
        results = []
        
        for method_name in traditional_methods:
            try:
                cache_key = f"{method_name}_{hash(str(df.values.tobytes()) + target)}"
                
                if cache_key in _cache['traditional_methods']:
                    res = _cache['traditional_methods'][cache_key]
                    res['cached'] = True
                    print(f"📦 استخدام النتيجة المخزنة للطريقة: {method_name}")
                else:
                    method_map = {
                        'embedding_rf': tm.embedding_rf,
                        'l1_logistic': tm.l1_logistic,
                        'rfe_rf': tm.rfe_rf
                    }
                    func = method_map.get(method_name)
                    print(f"🔄 تشغيل الطريقة: {method_name}")
                    res = func(df, target)
                    if 'method' not in res:
                        res['method'] = method_name
                    
                    # تحويل النتائج إلى أنواع قابلة للتحويل إلى JSON
                    res = _convert_to_serializable(res)
                    
                    _cache['traditional_methods'][cache_key] = res
                    res['cached'] = False
                    print(f"✅ تم تشغيل الطريقة: {method_name}")
                
                results.append(res)
            except Exception as e:
                print(f"❌ خطأ في {method_name}: {str(e)}")
                results.append({'method': method_name, 'error': str(e), 'selected_features': []})
        
        print("✅ تم تشغيل جميع الطرق التقليدية")
        return jsonify({'methods': results})
    except Exception as e:
        print(f"❌ خطأ في تشغيل الطرق التقليدية: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'خطأ في تشغيل الطرق التقليدية: ' + str(e)}), 500

@app.route('/api/run_all_statistical', methods=['POST'])
def api_run_all_statistical():
    """تشغيل جميع الطرق الإحصائية دفعة واحدة"""
    try:
        payload = request.json or {}
        print("🔍 تشغيل جميع الطرق الإحصائية...")
        
        df, target = _read_df_from_payload(payload)
        
        statistical_methods = ['filter_chi2', 'mutual_info', 'f_classif', 'variance_threshold']
        results = []
        
        for method_name in statistical_methods:
            try:
                cache_key = f"{method_name}_{hash(str(df.values.tobytes()) + target)}"
                
                if cache_key in _cache['traditional_methods']:
                    res = _cache['traditional_methods'][cache_key]
                    res['cached'] = True
                    print(f"📦 استخدام النتيجة المخزنة للطريقة: {method_name}")
                else:
                    method_map = {
                        'filter_chi2': tm.filter_chi2,
                        'mutual_info': tm.mutual_info,
                        'f_classif': tm.f_classif_filter,
                        'variance_threshold': tm.variance_threshold
                    }
                    func = method_map.get(method_name)
                    print(f"🔄 تشغيل الطريقة: {method_name}")
                    res = func(df, target)
                    if 'method' not in res:
                        res['method'] = method_name
                    
                    # تحويل النتائج إلى أنواع قابلة للتحويل إلى JSON
                    res = _convert_to_serializable(res)
                    
                    _cache['traditional_methods'][cache_key] = res
                    res['cached'] = False
                    print(f"✅ تم تشغيل الطريقة: {method_name}")
                
                results.append(res)
            except Exception as e:
                print(f"❌ خطأ في {method_name}: {str(e)}")
                results.append({'method': method_name, 'error': str(e), 'selected_features': []})
        
        print("✅ تم تشغيل جميع الطرق الإحصائية")
        return jsonify({'methods': results})
    except Exception as e:
        print(f"❌ خطأ في تشغيل الطرق الإحصائية: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'خطأ في تشغيل الطرق الإحصائية: ' + str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def api_compare():
    """مقارنة نتائج جميع الطرق"""
    try:
        payload = request.json or {}
        print("🔍 بدء المقارنة...")
        
        df, target = _read_df_from_payload(payload)

        methods_results = []

        provided = payload.get('methods', []) or []
        for pr in provided:
            methods_results.append(pr)

        include_ga_cached = payload.get('include_ga_cached', True)
        if include_ga_cached and _cache.get('last_ga') is not None:
            methods_results = [m for m in methods_results if not (m.get('method') == 'genetic' and m.get('cached') is not None)]
            ga_cached = dict(_cache['last_ga'])
            ga_cached['method'] = ga_cached.get('method', 'genetic')
            ga_cached['cached'] = True
            methods_results.append(ga_cached)

        stats = compare_and_stats(df, target, methods_results)
        plots = plot_results_base64(df, target, methods_results)
        
        # تحويل الإحصائيات إلى أنواع قابلة للتحويل إلى JSON
        stats = _convert_to_serializable(stats)
        
        print("✅ تمت المقارنة بنجاح")
        return jsonify({
            'stats': stats, 
            'plots': plots, 
            'methods_used': [m.get('method') for m in methods_results]
        })
    except Exception as e:
        print(f"❌ خطأ في المقارنة: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'خطأ في المقارنة: ' + str(e)}), 500

@app.route('/api/cache/status', methods=['GET'])
def api_cache_status():
    """الحصول على حالة الذاكرة المؤقتة"""
    return jsonify({
        'has_ga': _cache.get('last_ga') is not None, 
        'ga_params': _cache.get('last_ga_params'),
        'traditional_methods_count': len(_cache.get('traditional_methods', {}))
    })

@app.route('/api/cache/clear_all', methods=['POST'])
def api_clear_all_cache():
    """مسح كل الذاكرة المؤقتة"""
    try:
        _cache['last_ga'] = None
        _cache['last_ga_params'] = {}
        _cache['traditional_methods'] = {}
        return jsonify({'success': True, 'message': 'تم مسح كل الذاكرة المؤقتة'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# إضافة routes جديدة لعرض النتائج الفردية
@app.route('/api/results/traditional', methods=['GET'])
def api_get_traditional_results():
    """الحصول على نتائج الطرق التقليدية المخزنة"""
    try:
        traditional_methods = ['embedding_rf', 'l1_logistic', 'rfe_rf']
        results = []
        
        for method_name in traditional_methods:
            # البحث عن النتيجة في الذاكرة المؤقتة
            found = False
            for cache_key, cached_result in _cache['traditional_methods'].items():
                if method_name in cache_key:
                    results.append(cached_result)
                    found = True
                    break
            
            if not found:
                results.append({
                    'method': method_name, 
                    'selected_features': [], 
                    'cv_score': None,
                    'status': 'لم يتم التشغيل'
                })
        
        return jsonify({'methods': results})
    except Exception as e:
        print(f"❌ خطأ في الحصول على نتائج الطرق التقليدية: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/statistical', methods=['GET'])
def api_get_statistical_results():
    """الحصول على نتائج الطرق الإحصائية المخزنة"""
    try:
        statistical_methods = ['filter_chi2', 'mutual_info', 'f_classif', 'variance_threshold']
        results = []
        
        for method_name in statistical_methods:
            # البحث عن النتيجة في الذاكرة المؤقتة
            found = False
            for cache_key, cached_result in _cache['traditional_methods'].items():
                if method_name in cache_key:
                    results.append(cached_result)
                    found = True
                    break
            
            if not found:
                results.append({
                    'method': method_name, 
                    'selected_features': [], 
                    'cv_score': None,
                    'status': 'لم يتم التشغيل'
                })
        
        return jsonify({'methods': results})
    except Exception as e:
        print(f"❌ خطأ في الحصول على نتائج الطرق الإحصائية: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/ga', methods=['GET'])
def api_get_ga_results():
    """الحصول على نتائج الخوارزمية الجينية المخزنة"""
    try:
        if _cache.get('last_ga') is not None:
            return jsonify({'ga_result': _cache['last_ga']})
        else:
            return jsonify({'ga_result': None, 'message': 'لا توجد نتائج مخزنة'})
    except Exception as e:
        print(f"❌ خطأ في الحصول على نتائج GA: {str(e)}")
        return jsonify({'error': str(e)}), 500

# إضافة route للتحقق من صحة السيرفر
@app.route('/api/health', methods=['GET'])
def api_health():
    """التحقق من صحة السيرفر"""
    return jsonify({'status': 'ok', 'message': 'السيرفر يعمل بشكل صحيح'})

if __name__ == '__main__':
    print("🚀 بدء تشغيل السيرفر...")
    app.run(debug=True, host='0.0.0.0', port=5000)