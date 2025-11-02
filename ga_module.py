import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def individual(n_features, p_select=0.2):
    """إنشاء كروموسوم عشوائي (فرد في المجتمع)"""
    return [1 if random.random() < p_select else 0 for _ in range(n_features)]

def fitness(chromosome, X, y):
    """دالة اللياقة: تقييم جودة الكروموسوم"""
    mask = np.array(chromosome, dtype=bool)
    if mask.sum() == 0:
        return 0.0  # عقوبة إذا لم يتم اختيار أي ميزات
    
    X_sub = X.iloc[:, mask]
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    try:
        score = float(np.mean(cross_val_score(model, X_sub, y, cv=5, scoring='accuracy')))
    except Exception:
        return 0.0
    
    # عقوبة على عدد الميزات المختارة (لتجنب الإفراط في الاختيار)
    penalty = 0.01 * (mask.sum() / X.shape[1])
    return score - penalty

def tournament_selection(pop, fitnesses, k=3):
    """اختيار الأفراد باستخدام طريقة البطولة"""
    selected_idx = random.sample(range(len(pop)), k)
    best = max(selected_idx, key=lambda i: fitnesses[i])
    return pop[best]

def single_point_crossover(a, b):
    """تهجين بنقطة واحدة بين كروموسومين"""
    n = len(a)
    if n < 2:
        return a[:], b[:]
    pt = random.randint(1, n-1)
    child1 = a[:pt] + b[pt:]
    child2 = b[:pt] + a[pt:]
    return child1, child2

def mutate(chrom, rate=0.01):
    """تطبيق الطفرة على الكروموسوم"""
    return [1 - g if random.random() < rate else g for g in chrom]

def _chromosome_to_string(chrom):
    """تحويل الكروموسوم إلى سلسلة نصية"""
    return ''.join(str(int(bit)) for bit in chrom)

def run_genetic_algorithm(df, target_name, pop_size=40, generations=30, crossover_rate=0.8, mutation_rate=0.02, verbose=True):
    """تشغيل الخوارزمية الجينية لاختيار الميزات"""
    
    # التأكد من وجود العمود الهدف
    if target_name not in df.columns:
        target_name = df.columns[-1]

    X = df.drop(columns=[target_name])
    y = df[target_name]
    
    # تحويل المتغير الهدف إلى أعداد صحيحة
    try:
        y = y.astype(int)
    except Exception:
        y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)

    n_features = X.shape[1]

    # تهيئة المجتمع الأولي
    pop = [individual(n_features, p_select=0.2) for _ in range(pop_size)]
    best_solution = None
    best_fitness = -1.0
    history = []  # سجل تطور الأجيال

    if verbose:
        print(f"[GA] بدء التشغيل: المجتمع={pop_size}, الأجيال={generations}, الميزات={n_features}")

    for gen in range(generations):
        # حساب اللياقة لكل فرد في المجتمع
        fitnesses = [fitness(ind, X, y) for ind in pop]

        # العثور على أفضل فرد في الجيل الحالي
        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_f = float(fitnesses[gen_best_idx])
        gen_best = pop[gen_best_idx]
        gen_selected_count = int(sum(gen_best))

        # تسجيل تاريخ الجيل
        history.append({
            'generation': gen, 
            'best_fitness': gen_best_f, 
            'selected_count': gen_selected_count
        })

        # تحديث أفضل حل عالمي
        if gen_best_f > best_fitness:
            best_fitness = gen_best_f
            best_solution = gen_best.copy()

        if verbose:
            best_str = _chromosome_to_string(gen_best)
            print(f"[GA] الجيل {gen:03d} | اللياقة={gen_best_f:.4f} | المختارة={gen_selected_count}")

        # إنشاء الجيل الجديد
        new_pop = []
        while len(new_pop) < pop_size:
            # اختيار الآباء
            p1 = tournament_selection(pop, fitnesses)
            p2 = tournament_selection(pop, fitnesses)
            
            # التهجين
            if random.random() < crossover_rate:
                c1, c2 = single_point_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            
            # الطفرة
            c1 = mutate(c1, mutation_rate)
            c2 = mutate(c2, mutation_rate)
            
            new_pop.extend([c1, c2])
        
        pop = new_pop[:pop_size]

    # التأكد من وجود أفضل حل
    if best_solution is None:
        best_solution = pop[0]

    # حساب النتيجة النهائية
    final_score = float(fitness(best_solution, X, y))
    best_selected_count = int(sum(best_solution))
    best_chromosome_str = _chromosome_to_string(best_solution)
    selected_features = [X.columns[i] for i, bit in enumerate(best_solution) if bit]

    if verbose:
        print("--------------------------------------------------")
        print(f"[GA] الانتهاء. أفضل لياقة نهائية = {final_score:.6f}")
        print(f"[GA] عدد الميزات المختارة = {best_selected_count}")
        print("--------------------------------------------------")

    # إرجاع النتائج
    return {
        'method': 'genetic',
        'selected_features': selected_features,
        'final_score': final_score,
        'history': history,
        'best_chromosome': best_solution,
        'best_chromosome_str': best_chromosome_str,
        'best_selected_count': best_selected_count
    }